using Checkers.Common;
using Checkers.Core;
using Cysharp.Threading.Tasks;
using System;
using System.IO;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using UnityEngine;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using System.Collections.Generic;

namespace Checkers.Services
{
    public class ReinforcementLearningAIService : IAIService, IDisposable
    {
        private readonly string _host;
        private readonly int _port;
        private TcpClient _client;
        private NetworkStream _stream;
        private StreamReader _reader;
        private StreamWriter _writer;
        private readonly SemaphoreSlim _sendLock = new SemaphoreSlim(1, 1);
        private readonly int _readTimeoutMs = 5000;
        private readonly IGameRulesService _rulesService;

        private string _episodeId;

        public ReinforcementLearningAIService(string host, int port, IGameRulesService rulesService)
        {
            _host = host;
            _port = port;
            _rulesService = rulesService;
        }

        private async UniTask ConnectAsync()
        {
            if (_client != null && _client.Connected) return;

            SafeClose();

            _client = new TcpClient();
            await _client.ConnectAsync(_host, _port);
            _stream = _client.GetStream();
            _reader = new StreamReader(_stream, Encoding.UTF8);
            _writer = new StreamWriter(_stream, Encoding.UTF8) { AutoFlush = true };
            Debug.Log("Соединение с сервером установлено");
        }

        public async UniTask StartEpisodeAsync(BoardModel board, PlayerType player)
        {
            var response = await SendMessageAsync("start_episode", board, player);
            if (!string.IsNullOrEmpty(response))
            {
                try
                {
                    var jo = JObject.Parse(response);
                    var ep = jo["episode_id"]?.ToString();
                    if (!string.IsNullOrEmpty(ep))
                    {
                        _episodeId = ep;
                        Debug.Log($"Получено сообщение от сервера: {_episodeId}");
                    }
                }
                catch (Exception e)
                {
                    Debug.LogWarning($"Не получилось спарсить: {e} | ответ сервера: {response}");
                }
            }
        }

        public async UniTask EndEpisodeAsync(BoardModel board, PlayerType player, PlayerType? winner = null)
        {
            await SendMessageAsync("end_episode", board, player, winner);
            _episodeId = null;
        }

        public async UniTask<Move> GetMoveAsync(BoardModel board, PlayerType player)
        {
            var json = await SendMessageAsync("get_move", board, player);
            if (string.IsNullOrEmpty(json)) return null;

            await UniTask.Delay(1000);

            return DeserializeMove(json);
        }

        private async UniTask<string> SendMessageAsync(string type, BoardModel board, PlayerType player, PlayerType? winner = null)
        {
            await _sendLock.WaitAsync();
            try
            {
                await ConnectAsync();

                var dict = new Dictionary<string, object>
                {
                    ["type"] = type,
                    ["player"] = player.ToString(),
                    ["state"] = SerializeBoard(board, player)
                };

                if (winner.HasValue)
                {
                    dict["winner"] = winner.Value.ToString();
                }

                if (!string.IsNullOrEmpty(_episodeId))
                {
                    dict["episode_id"] = _episodeId;
                }

                string json = JsonConvert.SerializeObject(dict);
                await _writer.WriteLineAsync(json);

                using var cts = new CancellationTokenSource(_readTimeoutMs);
                try
                {
                    string response = await _reader.ReadLineAsync().AsUniTask().AttachExternalCancellation(cts.Token);
                    if (!string.IsNullOrEmpty(response))
                        Debug.Log($"Получено сообщение (длина {response.Length}) для {type}");
                    return response;
                }
                catch (OperationCanceledException)
                {
                    Debug.LogWarning("Таймаут");
                    return null;
                }
                catch (Exception e)
                {
                    Debug.LogWarning($"Ошибка: {e}");
                    SafeClose();
                    return null;
                }
            }
            catch (Exception e)
            {
                Debug.LogWarning($"Ошибка: {e}");
                SafeClose();
                return null;
            }
            finally
            {
                _sendLock.Release();
            }
        }

        private void SafeClose()
        {
            try { _reader?.Dispose(); } catch { }
            try { _writer?.Dispose(); } catch { }
            try { _stream?.Dispose(); } catch { }
            try { _client?.Close(); } catch { }
            _client = null;
            _episodeId = null;
        }

        private string SerializeBoard(BoardModel board, PlayerType player)
        {
            var pieces = new List<object>();
            foreach (var piece in board.GetAllPieces())
            {
                pieces.Add(new
                {
                    row = piece.Position.Row,
                    col = piece.Position.Column,
                    isKing = piece.IsKing,
                    owner = piece.Owner.ToString()
                });
            }

            var legalMoves = new List<object>();
            var moves = _rulesService.GetValidMoves(board, player);
            foreach (var m in moves)
            {
                legalMoves.Add(new
                {
                    fromRow = m.From.Row,
                    fromCol = m.From.Column,
                    toRow = m.To.Row,
                    toCol = m.To.Column,
                    captured = m.CapturedPieces != null
                        ? Array.ConvertAll(m.CapturedPieces, p => new int[] { p.Row, p.Column })
                        : new int[0][]
                });
            }

            var data = new { pieces, legal_moves = legalMoves };
            return JsonConvert.SerializeObject(data);
        }

        private Move DeserializeMove(string json)
        {
            if (string.IsNullOrEmpty(json)) return null;

            try
            {
                var obj = JsonConvert.DeserializeObject<MoveDTO>(json);
                if (obj == null) return null;

                return new Move(
                    new BoardPosition(obj.fromRow, obj.fromCol),
                    new BoardPosition(obj.toRow, obj.toCol),
                    obj.captured != null ? obj.captured.ConvertAll(p => new BoardPosition(p[0], p[1])).ToArray() : new BoardPosition[0]
                );
            }
            catch (Exception e)
            {
                Debug.LogWarning($"Неполучилось разобрать ход: {e} | ответ сервера: {json}");
                return null;
            }
        }

        [Serializable]
        private class MoveDTO
        {
            public int fromRow;
            public int fromCol;
            public int toRow;
            public int toCol;
            public System.Collections.Generic.List<int[]> captured;
        }

        public void Dispose() => SafeClose();
    }
}