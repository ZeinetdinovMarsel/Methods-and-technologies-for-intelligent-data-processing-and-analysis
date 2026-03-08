import socketserver
import json
import torch
import torch.nn as nn
import numpy as np
import uuid
import os

BOARD_SIZE = 8
NUM_CELLS = BOARD_SIZE * BOARD_SIZE
ACTION_SPACE = NUM_CELLS * NUM_CELLS
INPUT_CHANNELS = 5
INPUT_DIM = NUM_CELLS * INPUT_CHANNELS

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "checkpoints/ppo_16894.pth"

class Policy(nn.Module):
    def __init__(self, input_dim=INPUT_DIM, hidden1=512, hidden2=512, hidden3=256, output_dim=ACTION_SPACE):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden1), nn.ReLU(),
            nn.Linear(hidden1, hidden2), nn.ReLU(),
            nn.Linear(hidden2, hidden3), nn.ReLU()
        )
        self.actor = nn.Linear(hidden3, output_dim)
        self.critic = nn.Linear(hidden3, 1)

    def forward(self, x):
        h = self.shared(x)
        logits = self.actor(h)
        value = self.critic(h).squeeze(-1)
        return logits, value

def encode_board(state_json):
    arr = np.zeros(INPUT_DIM, dtype=np.float32)
    c0 = 0; c1 = NUM_CELLS; c2 = NUM_CELLS*2; c3 = NUM_CELLS*3; c4 = NUM_CELLS*4
    pieces = state_json.get("pieces", []) or []
    for p in pieces:
        r = int(p.get("row", 0)); c = int(p.get("col", 0))
        if 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE:
            idx = r * BOARD_SIZE + c
            owner = p.get("owner","White"); isKing = bool(p.get("isKing",False))
            if owner=="White": arr[c1+idx if isKing else c0+idx]=1.0
            else: arr[c3+idx if isKing else c2+idx]=1.0
    player = state_json.get("player","White")
    arr[c4:c4+NUM_CELLS] = 1.0 if player=="White" else -1.0
    return arr

def move_to_action_index(move):
    fr, fc = int(move["fromRow"]), int(move["fromCol"])
    tr, tc = int(move["toRow"]), int(move["toCol"])
    return (fr*BOARD_SIZE+fc)*NUM_CELLS + (tr*BOARD_SIZE+tc)

def action_index_to_move(ai):
    from_idx = ai // NUM_CELLS; to_idx = ai % NUM_CELLS
    fr, fc = from_idx//BOARD_SIZE, from_idx%BOARD_SIZE
    tr, tc = to_idx//BOARD_SIZE, to_idx%BOARD_SIZE
    return {"fromRow":fr,"fromCol":fc,"toRow":tr,"toCol":tc,"captured":[]}

class RLAgent:
    def __init__(self, model_path=MODEL_PATH, device=DEVICE, stochastic=False):
        self.device = device
        self.policy = Policy().to(device)
        if os.path.exists(model_path):
            self.policy.load_state_dict(torch.load(model_path,map_location=device))
            print("Eval: Loaded model:", model_path)
        else:
            latest_file = os.path.join("checkpoints","latest.txt")
            if os.path.exists(latest_file):
                with open(latest_file,'r') as f:
                    name = f.read().strip()
                candidate = os.path.join("checkpoints", name)
                if os.path.exists(candidate):
                    self.policy.load_state_dict(torch.load(candidate,map_location=device))
                    print("Eval: Loaded latest:", candidate)
        self.policy.eval()
        self.stochastic = stochastic

    def select_move(self, state_json, legal_moves):
        state_np = encode_board(state_json)
        state = torch.from_numpy(state_np).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits,_ = self.policy(state)
        logits = logits.squeeze(0)

        mask = np.zeros(ACTION_SPACE, dtype=bool)
        for mv in legal_moves:
            try: mask[move_to_action_index(mv)] = True
            except: pass
        if not mask.any(): mask[0]=True

        mask_t = torch.from_numpy(mask).to(self.device)
        masked_logits = logits.clone()
        masked_logits[~mask_t] = -1e9

        if self.stochastic:
            dist = torch.distributions.Categorical(logits=masked_logits)
            ai = int(dist.sample().item())
        else:
            ai = int(masked_logits.argmax().item())

        move = next((mv for mv in legal_moves if move_to_action_index(mv)==ai),None)
        if move is None: move = action_index_to_move(ai)
        return move

agent = RLAgent(stochastic=False)

class ThreadedTCPRequestHandler(socketserver.BaseRequestHandler):
    episode_counter = 0
    win_counters = {"White":0,"Black":0,"Draw":0}

    def handle(self):
        rfile = self.request.makefile('rb')
        wfile = self.request.makefile('wb')
        current_episode_id = None

        for raw_line in rfile:
            try:
                data = raw_line.decode('utf-8-sig').strip()
                if not data: continue
                msg = json.loads(data)

                msg_type = msg.get("type")
                player = msg.get("player","White")
                state_field = msg.get("state","{}")
                state_json = json.loads(state_field) if isinstance(state_field,str) else state_field
                state_json["player"] = player

                ep_id = msg.get("episode_id") or current_episode_id
                current_episode_id = ep_id

                resp = {"error":"unknown_type"}

                if msg_type == "start_episode":
                    ep_id = str(uuid.uuid4())
                    current_episode_id = ep_id
                    resp = {"status":"ok","episode_id":ep_id}
                    print("Начат эпизод:", ep_id)

                elif msg_type == "get_move":
                    legal_moves = state_json.get("legal_moves", []) or []
                    if legal_moves:
                        move = agent.select_move(state_json, legal_moves)
                        resp = move
                    else:
                        resp = {"error":"no_moves"}

                elif msg_type == "end_episode":
                    winner = msg.get("winner")
                    ThreadedTCPRequestHandler.episode_counter += 1
                    if winner in ThreadedTCPRequestHandler.win_counters:
                        ThreadedTCPRequestHandler.win_counters[winner] += 1

                    total = ThreadedTCPRequestHandler.episode_counter
                    print(f"[Эпизод {total}] Конец - Победитель: {winner} | Статистика W/B/D: "
                          f"{ThreadedTCPRequestHandler.win_counters['White']}/{ThreadedTCPRequestHandler.win_counters['Black']}/{ThreadedTCPRequestHandler.win_counters['Draw']}")
                    resp = {"status":"ok"}

                wfile.write((json.dumps(resp)+"\n").encode('utf-8'))
                wfile.flush()

            except Exception as e:
                try: wfile.write((json.dumps({"error":str(e)})+"\n").encode('utf-8'))
                except: pass

class ThreadedTCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    daemon_threads = True
    allow_reuse_address = True

if __name__=="__main__":
    HOST, PORT = "127.0.0.1", 5556
    print(f"Сервер тестирования запущен {HOST}:{PORT}, устройство={DEVICE}")
    with ThreadedTCPServer((HOST,PORT), ThreadedTCPRequestHandler) as server:
        server.serve_forever()