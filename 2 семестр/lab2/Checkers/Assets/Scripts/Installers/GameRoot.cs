using Checkers.Common;
using Checkers.Core;
using Checkers.Services;
using Checkers.Views;
using Cysharp.Threading.Tasks;
using System;
using System.Linq;
using System.Threading;
using UniRx;
using UnityEngine;
using Zenject;

namespace Checkers.Installers
{
    public class GameRoot : MonoBehaviour
    {
        [Inject] private GameModel _gameModel;
        [Inject] private BoardModel _boardModel;
        [Inject] private IGameService _gameService;
        [Inject] private IGameRulesService _rulesService;
        [Inject] private BoardView _boardView;
        [Inject] private UIView _uiView;

        private readonly CompositeDisposable _disposables = new();
        private BoardPosition? _selectedPosition;
        [SerializeField] private bool _autoRestart;

        private CancellationTokenSource _turnCts;

        private void Start()
        {
            InitializeGame();
            SetupBindings();
        }

        private void InitializeGame()
        {
            _boardView.Initialize();
            _gameService.Initialize();
            _uiView.Bind(_gameModel);
            SyncBoardView();

            StartTurnLoop();
        }

        private void StartTurnLoop()
        {
            _turnCts?.Cancel();
            _turnCts?.Dispose();
            _turnCts = new CancellationTokenSource();
        }

        private void RestartGame()
        {
            _boardView.ClearBoard();
            _selectedPosition = null;

            _gameService.ResetGame();
            _gameService.Initialize();

            SyncBoardView();

            _gameService.HandleTurnAsync();




            StartTurnLoop();
        }

        private void SetupBindings()
        {
            _boardView.OnCellClicked
                .Subscribe(OnCellClicked)
                .AddTo(_disposables);

            _uiView.OnRestartClicked
                .Subscribe(_ => RestartGame())
                .AddTo(_disposables);

            _uiView.OnQuitClicked
                .Subscribe(_ => Application.Quit())
                .AddTo(_disposables);

            _gameModel.LastMove
                .Where(move => move != null)
                .Subscribe(OnMoveMade)
                .AddTo(_disposables);

            _gameModel.LastMove
                .Where(move => move != null)
                .Subscribe(move => _boardView.HighlightLastMove(move))
                .AddTo(_disposables);

            _gameModel.CurrentTurn
                .Subscribe(_ => OnTurnChanged().Forget())
                .AddTo(_disposables);

            _gameModel.State
                .Where(state => state == GameState.GameOver && _autoRestart)
                .Subscribe(_ =>
                {
                    var winner = _gameModel.Winner.Value;
                    _uiView.ShowGameOver(winner);

                    RestartGame();
                })
                .AddTo(_disposables);
        }

        private void OnMoveMade(Move move)
        {
            _boardView.MovePiece(move);

            foreach (var capturedPos in move.CapturedPieces)
            {
                _boardView.RemovePiece(capturedPos);
            }

            _boardView.HighlightCells(new BoardPosition[0]);
            _selectedPosition = null;
        }

        private void OnCellClicked(BoardPosition position)
        {
            if (_gameModel.State.Value != GameState.Playing) return;
            var control = _gameService.GetCurrentPlayerControl();

            if (control != PlayerControlType.Human) return;

            var piece = _boardModel.GetPiece(position);
            if (_selectedPosition.HasValue)
            {
                var validMove = FindValidMove(_selectedPosition.Value, position);

                if (validMove != null)
                {
                    _gameService.MakeMove(validMove);
                    return;
                }
            }

            if (piece != null && piece.Owner == _gameModel.CurrentTurn.Value)
            {
                var moves = _rulesService.GetValidMovesForPiece(_boardModel, piece);
                if (moves.Length == 0) return;

                _selectedPosition = position;
                HighlightValidMoves(position);
            }
            else
            {
                _selectedPosition = null;
                _boardView.HighlightCells(new BoardPosition[0]);
            }
        }

        private Move FindValidMove(BoardPosition from, BoardPosition to)
        {
            var piece = _boardModel.GetPiece(from);
            if (piece == null) return null;

            var validMoves = _rulesService.GetValidMovesForPiece(_boardModel, piece);

            foreach (var move in validMoves)
            {
                if (move.From == from && move.To == to)
                {
                    return move;
                }
            }

            return null;
        }

        private void HighlightValidMoves(BoardPosition position)
        {
            var piece = _boardModel.GetPiece(position);
            if (piece == null) return;

            var validMoves = _rulesService.GetValidMovesForPiece(_boardModel, piece);

            var targetPositions = validMoves
                .Select(m => m.To)
                .ToArray();

            _boardView.HighlightCells(targetPositions);
        }

        private async UniTask OnTurnChanged()
        {
            if (_gameModel.State.Value != GameState.Playing) return;

            await _gameService.HandleTurnAsync();
        }

        private void SyncBoardView()
        {
            for (int row = 0; row < GameConstants.BoardSize; row++)
            {
                for (int col = 0; col < GameConstants.BoardSize; col++)
                {
                    var position = new BoardPosition(row, col);
                    var piece = _boardModel.GetPiece(position);

                    if (piece != null)
                    {
                        _boardView.SpawnPiece(piece);
                    }
                }
            }
        }

        private void OnDestroy()
        {
            _disposables?.Dispose();
        }
    }
}