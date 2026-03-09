import socketserver
import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from torch.utils.tensorboard import SummaryWriter
from json import JSONDecodeError
import time
import threading
import uuid

BOARD_SIZE = 8
NUM_CELLS = BOARD_SIZE * BOARD_SIZE
ACTION_SPACE = NUM_CELLS * NUM_CELLS
INPUT_CHANNELS = 5
INPUT_DIM = NUM_CELLS * INPUT_CHANNELS

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

GAMMA = 0.995
LAMBDA = 0.95
CLIP_EPS = 0.2
LR = 1e-4
VALUE_COEF = 0.5
ENTROPY_COEF = 0.005

UPDATE_EPOCHS = 6
MINIBATCH_SIZE = 256
ROLLOUT_SIZE = 4096

CHECKPOINT_DIR = "checkpoints"
RUN_DIR = "runs/checkers_safe"

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(RUN_DIR, exist_ok=True)

capture_piece_reward = 0.3
king_piece_reward = 0.5
win_reward = 5
lose_punishment = -5

writer = SummaryWriter(RUN_DIR)
torch.cuda.manual_seed_all(0)
np.random.seed(0)

class Policy(nn.Module):
    def __init__(self, input_dim=INPUT_DIM, hidden1=512, hidden2=512, hidden3=256, output_dim=ACTION_SPACE):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, hidden3),
            nn.ReLU(),
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
    c0 = 0
    c1 = NUM_CELLS
    c2 = NUM_CELLS * 2
    c3 = NUM_CELLS * 3
    c4 = NUM_CELLS * 4
    pieces = state_json.get("pieces", []) or []
    for p in pieces:
        r = int(p.get("row", 0))
        c = int(p.get("col", 0))
        if 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE:
            idx = r * BOARD_SIZE + c
            owner = p.get("owner", "White")
            isKing = bool(p.get("isKing", False))
            if owner == "White":
                if isKing:
                    arr[c1 + idx] = 1.0
                else:
                    arr[c0 + idx] = 1.0
            else:
                if isKing:
                    arr[c3 + idx] = 1.0
                else:
                    arr[c2 + idx] = 1.0
    player = state_json.get("player", "White")
    arr[c4:c4 + NUM_CELLS] = 1.0 if player == "White" else -1.0
    return arr

def move_to_action_index(move):
    fr = int(move["fromRow"]); fc = int(move["fromCol"])
    tr = int(move["toRow"]); tc = int(move["toCol"])
    return (fr * BOARD_SIZE + fc) * NUM_CELLS + (tr * BOARD_SIZE + tc)

def action_index_to_move(ai):
    from_idx = ai // NUM_CELLS
    to_idx = ai % NUM_CELLS
    fr = from_idx // BOARD_SIZE; fc = from_idx % BOARD_SIZE
    tr = to_idx // BOARD_SIZE; tc = to_idx % BOARD_SIZE
    return {"fromRow": fr, "fromCol": fc, "toRow": tr, "toCol": tc, "captured": []}

class RolloutBuffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.old_logprobs = []
        self.values = []
        self.rewards = []
        self.masks = []
        self.dones = []
        self.episode_ids = []

    def clear(self):
        self.__init__()

buffer = RolloutBuffer()
buffer_lock = threading.Lock()

pending_episode_rewards = {}
pending_lock = threading.Lock()

update_in_progress = False
save_lock = threading.Lock()

class PPOAgent:
    def __init__(self, device=DEVICE):
        self.device = device
        self.policy = Policy().to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=LR)
        self.step = 0
        self.load_latest_checkpoint_if_any()
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lambda step: 1 - step / 1_000_000
        )

    def select_action(self, state_np, legal_moves):
        state = torch.from_numpy(state_np).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits, value = self.policy(state)
        logits = logits.squeeze(0)
        value = float(value.item())

        mask = np.zeros(ACTION_SPACE, dtype=np.bool_)
        for mv in legal_moves:
            try:
                ai = move_to_action_index(mv)
                if 0 <= ai < ACTION_SPACE:
                    mask[ai] = True
            except:
                pass
        if not mask.any():
            if legal_moves:
                mask[move_to_action_index(legal_moves[0])] = True
            else:
                mask[0] = True

        mask_t = torch.from_numpy(mask).to(self.device)
        inf_neg = -1e9
        masked_logits = logits.clone()
        masked_logits[~mask_t] = inf_neg

        dist = torch.distributions.Categorical(logits=masked_logits)
        action = int(dist.sample().item())
        logprob = float(dist.log_prob(torch.tensor(action, device=self.device)).item())

        chosen_move = next((mv for mv in legal_moves if move_to_action_index(mv) == action), None)
        if chosen_move is None:
            chosen_move = action_index_to_move(action)

        return chosen_move, int(action), logprob, value, mask.copy()

    def greedy_action(self, state_np, legal_moves):
        state = torch.from_numpy(state_np).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits, _ = self.policy(state)
        logits = logits.squeeze(0).cpu().numpy()
        best = None; best_val = -1e9
        for mv in legal_moves:
            try:
                ai = move_to_action_index(mv)
                val = logits[ai]
                if val > best_val:
                    best_val = val
                    best = mv
            except: pass
        if best is None:
            best = legal_moves[0] if legal_moves else action_index_to_move(0)
        return best

    def update_from_copy(self, states_np, actions_np, old_logprobs_np, values_np, rewards_np, masks_np, dones_np, episode_ids_np,
                         update_epochs=UPDATE_EPOCHS, minibatch_size=MINIBATCH_SIZE):
        try:
            states = torch.from_numpy(np.vstack([s.copy() for s in states_np])).float().to(self.device)
        except Exception as e:
            print("Ошибка обновления с копии:", e)
            return

        actions = torch.tensor(actions_np, dtype=torch.long, device=self.device)
        old_logprobs = torch.tensor(old_logprobs_np, dtype=torch.float32, device=self.device)

        try:
            masks = torch.from_numpy(np.vstack([m.copy() for m in masks_np]).astype(np.bool_)).to(self.device)
        except Exception as e:
            print("Ошибка в составлении маски с копии:", e)
            masks = torch.ones((states.shape[0], ACTION_SPACE), dtype=torch.bool, device=self.device)

        values = list(values_np)

        N = len(rewards_np)
        advantages = []
        gae = 0.0
        for t in reversed(range(N)):
            if t + 1 < N and episode_ids_np[t] == episode_ids_np[t + 1]:
                next_value = values[t + 1]
                next_same = True
            else:
                next_value = 0.0
                next_same = False

            delta = rewards_np[t] + GAMMA * next_value - values[t]
            if next_same:
                gae = delta + GAMMA * LAMBDA * gae
            else:
                gae = delta
            advantages.insert(0, gae)

        returns = [adv + v for adv, v in zip(advantages, values)]
        adv_t = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        returns_t = torch.tensor(returns, dtype=torch.float32, device=self.device)

        adv_t = (adv_t - adv_t.mean()) / (adv_t.std(unbiased=False) + 1e-8)

        entropy_coef = max(0.0005, ENTROPY_COEF * (0.999 ** self.step))
        total_batch = states.shape[0]

        for epoch in range(update_epochs):
            perm = torch.randperm(total_batch, device=self.device)
            for start in range(0, total_batch, minibatch_size):
                mb_idx = perm[start:start + minibatch_size]
                mb_states = states[mb_idx]
                mb_actions = actions[mb_idx]
                mb_old_logprobs = old_logprobs[mb_idx]
                mb_returns = returns_t[mb_idx]
                mb_adv = adv_t[mb_idx]
                mb_masks = masks[mb_idx]

                logits, values_pred = self.policy(mb_states)
                masked_logits = logits.clone()
                masked_logits[~mb_masks] = -1e9
                dist = torch.distributions.Categorical(logits=masked_logits)
                new_logprobs = dist.log_prob(mb_actions)
                entropy = dist.entropy().mean()
                ratio = torch.exp(new_logprobs - mb_old_logprobs)
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1.0 - CLIP_EPS, 1.0 + CLIP_EPS) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = VALUE_COEF * ((mb_returns - values_pred) ** 2).mean()
                loss = policy_loss + value_loss - entropy_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                self.optimizer.step()
                self.scheduler.step()

        writer.add_scalar('train/entropy_coef', entropy_coef, self.step)
        self.step += 1

    def save(self, tag=None):
        if tag is None: tag = int(time.time())
        path = os.path.join(CHECKPOINT_DIR, f'ppo_{tag}.pth')
        with save_lock:
            torch.save({
                'policy_state': self.policy.state_dict(),
                'optimizer_state': self.optimizer.state_dict(),
                'step': self.step
            }, path)
            with open(os.path.join(CHECKPOINT_DIR, 'latest.txt'), 'w') as f:
                f.write(os.path.basename(path))
        return path

    def load_latest_checkpoint_if_any(self):
        latest_file = os.path.join(CHECKPOINT_DIR, 'latest.txt')
        if os.path.exists(latest_file):
            try:
                with open(latest_file, 'r') as f:
                    name = f.read().strip()
                candidate = os.path.join(CHECKPOINT_DIR, name)
                if os.path.exists(candidate):
                    checkpoint = torch.load(candidate, map_location=self.device)
                    self.policy.load_state_dict(checkpoint['policy_state'])
                    self.optimizer.load_state_dict(checkpoint['optimizer_state'])
                    self.step = checkpoint.get('step', 0)
                    print("Загружен чекпоинт:", candidate)
            except Exception as e:
                print("Ошибка при загрузке чекпоинта:", e)

agent = PPOAgent(device=DEVICE)

class ThreadedTCPRequestHandler(socketserver.BaseRequestHandler):
    episode_counter = 0
    win_counters = {"White":0,"Black":0,"Draw":0}

    def handle(self):
        global update_in_progress
        self.current_episode_id = None

        rfile = self.request.makefile('rb')
        wfile = self.request.makefile('wb')

        for raw_line in rfile:
            try:
                data = raw_line.decode('utf-8-sig', errors='ignore').strip()
                if not data: continue
                try:
                    msg = json.loads(data)
                except JSONDecodeError:
                    idx = data.find('{')
                    if idx != -1:
                        msg = json.loads(data[idx:])
                    else:
                        continue

                msg_type = msg.get("type")
                player = msg.get("player","White")
                state_field = msg.get("state","{}")
                state_json = json.loads(state_field) if isinstance(state_field,str) else state_field
                state_json["player"] = player

                resp = {"error":"unknown_type"}

                received_ep_id = msg.get("episode_id")
                if received_ep_id is not None:
                    ep_id = received_ep_id
                    self.current_episode_id = ep_id
                else:
                    ep_id = self.current_episode_id

                if msg_type == "start_episode":
                    ep_id = str(uuid.uuid4())
                    self.current_episode_id = ep_id
                    with pending_lock:
                        pending_episode_rewards.setdefault(ep_id, pending_episode_rewards.get(ep_id, 0.0))
                    resp = {"status":"ok", "episode_id": ep_id}
                    print(f"Начат эпизод ep_id={ep_id}")

                elif msg_type == "get_move":
                    legal_moves = state_json.get("legal_moves", []) or []
                    if legal_moves:
                        if ep_id is None:
                            ep_id = str(uuid.uuid4())
                            self.current_episode_id = ep_id
                            with pending_lock:
                                pending_episode_rewards.setdefault(ep_id, 0.0)

                        state_np = encode_board(state_json)
                        chosen_move, action_idx, logprob, value, mask_np = agent.select_action(state_np, legal_moves)

                        captured = chosen_move.get("captured",[]) or []
                        step_reward = capture_piece_reward * len(captured)

                        if "kinged" in chosen_move and chosen_move.get("kinged"):
                            step_reward += king_piece_reward

                        with pending_lock:
                            pending = pending_episode_rewards.get(ep_id, 0.0)
                            if pending != 0.0:
                                step_reward += pending
                                pending_episode_rewards[ep_id] = 0.0

                        do_start_update = False
                        with buffer_lock:
                            buffer.states.append(state_np.copy())
                            buffer.actions.append(action_idx)
                            buffer.old_logprobs.append(logprob)
                            buffer.values.append(value)
                            buffer.rewards.append(step_reward)
                            buffer.masks.append(mask_np.copy())
                            buffer.dones.append(False)
                            buffer.episode_ids.append(ep_id)

                            with buffer_lock:
                                if len(buffer.states) >= ROLLOUT_SIZE and not update_in_progress:
                                    update_in_progress = True
                                    states_copy = [s.copy() for s in buffer.states]
                                    actions_copy = buffer.actions[:]
                                    old_logprobs_copy = buffer.old_logprobs[:]
                                    values_copy = buffer.values[:]
                                    rewards_copy = buffer.rewards[:]
                                    masks_copy = [m.copy() for m in buffer.masks]
                                    dones_copy = buffer.dones[:]
                                    episode_ids_copy = buffer.episode_ids[:]
                                    buffer.clear()
                                    do_start_update = True

                        if do_start_update:
                            def do_update(states_c, actions_c, old_lp_c, values_c, rewards_c, masks_c, dones_c, episode_ids_c):
                                global update_in_progress
                                pid = os.getpid()
                                print(f"Обновление запущено pid={pid} кол-во шагов={len(states_c)}")
                                try:
                                    agent.update_from_copy(states_c, actions_c, old_lp_c,
                                                           values_c, rewards_c, masks_c, dones_c, episode_ids_c)
                                    path = agent.save(tag=ThreadedTCPRequestHandler.episode_counter)
                                    print(f"Модель сохранена {path} (pid={pid})")
                                except Exception as e:
                                    print("Ошибка при обновлениии:", e)
                                finally:
                                    with buffer_lock:
                                        update_in_progress = False
                                        print(f"Обновление завершено pid={pid}")

                            threading.Thread(target=do_update, args=(states_copy, actions_copy, old_logprobs_copy,
                                                                     values_copy, rewards_copy, masks_copy,
                                                                     dones_copy, episode_ids_copy), daemon=True).start()

                        resp = chosen_move
                    else:
                        resp = {"error":"no_moves"}

                elif msg_type == "end_episode":
                    winner = msg.get("winner")
                    if ep_id is None:
                        ep_id = str(uuid.uuid4())

                    if winner == player:
                        reward_delta = win_reward
                    elif winner == "Draw":
                        reward_delta = 0.0
                    else:
                        reward_delta = -lose_punishment

                    attached = False
                    with buffer_lock:
                        for i in range(len(buffer.episode_ids) - 1, -1, -1):
                            if buffer.episode_ids[i] == ep_id:
                                buffer.rewards[i] += reward_delta
                                attached = True

                    if not attached:
                        with pending_lock:
                            pending_episode_rewards[ep_id] = pending_episode_rewards.get(ep_id, 0.0) + reward_delta

                    ThreadedTCPRequestHandler.episode_counter += 1
                    if winner:
                        ThreadedTCPRequestHandler.win_counters[winner] = ThreadedTCPRequestHandler.win_counters.get(winner,0)+1
                    total = ThreadedTCPRequestHandler.episode_counter
                    print(f"Эпизод {total} Конец - Победитель: {winner} (ep_id={ep_id})")
                    writer.add_scalar('game/episodes', total, total)
                    writer.add_scalar('game/win_white', ThreadedTCPRequestHandler.win_counters.get('White',0)/max(1,total), total)
                    writer.add_scalar('game/win_black', ThreadedTCPRequestHandler.win_counters.get('Black',0)/max(1,total), total)
                    writer.add_scalar('game/draw_rate', ThreadedTCPRequestHandler.win_counters.get('Draw',0)/max(1,total), total)
                    writer.flush()
                    resp = {"status":"ok"}

                elif msg_type == "save_model":
                    path = agent.save(tag=f"manual_{int(time.time())}")
                    resp = {"status":"saved","path":path}

                elif msg_type == "load_model":
                    agent.load_latest_checkpoint_if_any()
                    resp = {"status":"loaded"}

                try:
                    wfile.write((json.dumps(resp) + "\n").encode('utf-8'))
                    wfile.flush()
                except Exception as e:
                    print("Ошибка при попытке ответить клиенту:", e)

            except Exception as e:
                try:
                    wfile.write((json.dumps({"error":str(e)}) + "\n").encode('utf-8'))
                    wfile.flush()
                except:
                    pass

class ThreadedTCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    daemon_threads = True
    allow_reuse_address = True

def periodic_save(agent, interval_sec=300):
    while True:
        time.sleep(interval_sec)
        try:
            path = agent.save(tag=f"auto_{int(time.time())}")
            print(f"Модель автоматически сохранена {path}")
        except Exception as e:
            print("Ошибка в автосохранении", e)

threading.Thread(target=periodic_save, args=(agent,300), daemon=True).start()

if __name__ == "__main__":
    HOST, PORT = "127.0.0.1", 5555
    print(f"Сервер запушен {HOST}:{PORT}, устройство={DEVICE}, pid={os.getpid()}")
    with ThreadedTCPServer((HOST,PORT), ThreadedTCPRequestHandler) as server:
        server.serve_forever()