using Checkers.Common;
using Checkers.Core;
using Cysharp.Threading.Tasks;
using System;
using System.Threading;
using UniRx;
using Zenject;

namespace Checkers.Services
{
    public class GameService : IGameService, IDisposable
    {
        private readonly GameModel _gameModel;
        private readonly BoardModel _boardModel;
        private readonly IGameRulesService _rulesService;

        private readonly IAIService _whiteAI;
        private readonly IAIService _blackAI;

        private readonly CompositeDisposable _disposables = new();

        private PlayerControlType _whitePlayer;
        private PlayerControlType _blackPlayer;
        private CancellationTokenSource _cts;
        public GameService(
            GameModel gameModel,
            BoardModel boardModel,
            IGameRulesService rulesService,
            [InjectOptional(Id = PlayerType.White)] IAIService whiteAI,
            [InjectOptional(Id = PlayerType.Black)] IAIService blackAI)
        {
            _gameModel = gameModel;
            _boardModel = boardModel;
            _rulesService = rulesService;

            _whiteAI = whiteAI;
            _blackAI = blackAI;
        }

        public void SetPlayers(PlayerControlType white, PlayerControlType black)
        {
            _whitePlayer = white;
            _blackPlayer = black;
        }

        public void Initialize()
        {
            _cts?.Cancel();
            _cts?.Dispose();

            _cts = new CancellationTokenSource();

            SetupBoard();
            UpdatePieceCounts();

            _gameModel.CurrentTurn.Value = PlayerType.White;
            _gameModel.State.Value = GameState.Playing;

            if (_whiteAI is not null && _whiteAI is ReinforcementLearningAIService whiteRLService)
            {
                whiteRLService.StartEpisodeAsync(_boardModel, PlayerType.White).Forget();
            }

            if (_blackAI is not null && _blackAI is ReinforcementLearningAIService blackRLService)
            {
                blackRLService.StartEpisodeAsync(_boardModel, PlayerType.Black).Forget();
            }
        }

        public void ResetGame()
        {
            _cts?.Cancel();

            for (int row = 0; row < GameConstants.BoardSize; row++)
            {
                for (int col = 0; col < GameConstants.BoardSize; col++)
                {
                    _boardModel.SetPiece(new BoardPosition(row, col), null);
                }
            }

            if (_whiteAI is not null && _whiteAI is ReinforcementLearningAIService whiteRLService)
            {
                whiteRLService.EndEpisodeAsync(_boardModel, PlayerType.White, _gameModel.Winner.Value).Forget();
            }

            if (_blackAI is not null && _blackAI is ReinforcementLearningAIService blackRLService)
            {
                blackRLService.EndEpisodeAsync(_boardModel, PlayerType.Black, _gameModel.Winner.Value).Forget();
            }

            _gameModel.State.Value = GameState.NotStarted;
            _gameModel.CurrentTurn.Value = PlayerType.White;
            _gameModel.Winner.Value = PlayerType.None;
            _gameModel.LastMove.Value = null;
        }

        public void Dispose()
        {
            _cts?.Cancel();
            _cts?.Dispose();
            _disposables.Dispose();
        }

        private void SetupBoard()
        {
            for (int row = 0; row < GameConstants.BoardSize; row++)
            {
                for (int col = 0; col < GameConstants.BoardSize; col++)
                {
                    if (!_boardModel.IsBlackCell(new BoardPosition(row, col)))
                        continue;

                    if (row < 3)
                    {
                        _boardModel.SetPiece(
                            new BoardPosition(row, col),
                            new PieceModel(PlayerType.Black, new BoardPosition(row, col)));
                    }
                    else if (row > 4)
                    {
                        _boardModel.SetPiece(
                            new BoardPosition(row, col),
                            new PieceModel(PlayerType.White, new BoardPosition(row, col)));
                    }
                }
            }
        }

        public void MakeMove(Move move)
        {
            if (_gameModel.State.Value != GameState.Playing)
                return;

            var currentPlayer = _gameModel.CurrentTurn.Value;

            if (!_rulesService.IsValidMove(_boardModel, move, currentPlayer))
            {
                UnityEngine.Debug.LogWarning("Неверный ход!");
                return;
            }

            _rulesService.ApplyMove(_boardModel, move);

            _gameModel.LastMove.Value = move;

            UpdatePieceCounts();

            CheckGameOver(currentPlayer);

            if (_gameModel.State.Value == GameState.GameOver)
            {

                return;
            }

            _gameModel.CurrentTurn.Value =
                currentPlayer == PlayerType.White
                ? PlayerType.Black
                : PlayerType.White;
        }

        private void CheckGameOver(PlayerType currentPlayer)
        {
            if (_gameModel.WhitePiecesCount.Value == 0)
            {
                _gameModel.Winner.Value = PlayerType.Black;
                _gameModel.State.Value = GameState.GameOver;
                return;
            }

            if (_gameModel.BlackPiecesCount.Value == 0)
            {
                _gameModel.Winner.Value = PlayerType.White;
                _gameModel.State.Value = GameState.GameOver;
                return;
            }

            var next =
                currentPlayer == PlayerType.White
                               ? PlayerType.Black
                               : PlayerType.White;

            if (_rulesService.IsGameOver(_boardModel, next))
            {
                _gameModel.Winner.Value = currentPlayer;
                _gameModel.State.Value = GameState.GameOver;
            }
        }

        private IAIService GetAI(PlayerType player)
        {
            return player == PlayerType.White
                ? _whiteAI
                : _blackAI;
        }

        public PlayerControlType GetCurrentPlayerControl()
        {
            return _gameModel.CurrentTurn.Value == PlayerType.White
                ? _whitePlayer
                : _blackPlayer;
        }

        public async UniTask<Move> GetAIMoveAsync()
        {
            var player = _gameModel.CurrentTurn.Value;
            var ai = GetAI(player);

            if (ai == null)
                return null;

            return await ai.GetMoveAsync(_boardModel, player);
        }

        public async UniTask HandleTurnAsync()
        {
            if (_cts == null) _cts = new CancellationTokenSource();
            var token = _cts.Token;

            while (_gameModel.State.Value == GameState.Playing)
            {
                token.ThrowIfCancellationRequested();

                var control = GetCurrentPlayerControl();
                if (control != PlayerControlType.AI)
                    break;

                Move move = null;

                for (int i = 0; i < 10; i++)
                {
                    token.ThrowIfCancellationRequested();

                    move = await GetAIMoveAsync().AttachExternalCancellation(token);

                    if (move != null && _rulesService.IsValidMove(_boardModel, move, _gameModel.CurrentTurn.Value))
                        break;
                }

                if (move == null)
                    break;

                MakeMove(move);
            }
        }

        private void UpdatePieceCounts()
        {
            int white = 0;
            int black = 0;

            for (int row = 0; row < GameConstants.BoardSize; row++)
            {
                for (int col = 0; col < GameConstants.BoardSize; col++)
                {
                    var piece = _boardModel.GetPiece(new BoardPosition(row, col));

                    if (piece == null)
                        continue;

                    if (piece.Owner == PlayerType.White)
                        white++;
                    else
                        black++;
                }
            }

            _gameModel.WhitePiecesCount.Value = white;
            _gameModel.BlackPiecesCount.Value = black;
        }
    }
}