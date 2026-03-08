using System.Collections.Generic;
using System.Linq;
using UniRx;
using Checkers.Core;
using UnityEngine;
using System;
using Checkers.Common;
using Cysharp.Threading.Tasks;

namespace Checkers.Views
{
    public class BoardView : MonoBehaviour
    {
        [SerializeField] private CellView _cellPrefab;
        [SerializeField] private PieceView _piecePrefab;
        [SerializeField] private Transform _boardContainer;
        [SerializeField] private float _cellSize = 1f;

        private readonly Dictionary<BoardPosition, CellView> _cells = new();
        private readonly Dictionary<BoardPosition, PieceView> _pieces = new();

        public IObservable<BoardPosition> OnCellClicked { get; private set; }

        public void Initialize()
        {
            var clickObservables = new List<IObservable<BoardPosition>>();

            for (int row = 0; row < GameConstants.BoardSize; row++)
            {
                for (int col = 0; col < GameConstants.BoardSize; col++)
                {
                    var position = new BoardPosition(row, col);
                    var isBlackCell = (row + col) % 2 == 1;

                    var cell = Instantiate(_cellPrefab, _boardContainer);
                    cell.Initialize(position, isBlackCell);
                    cell.transform.localPosition = new Vector3(
                        col * _cellSize - (GameConstants.BoardSize * _cellSize) / 2 + _cellSize / 2,
                        0,
                        row * _cellSize - (GameConstants.BoardSize * _cellSize) / 2 + _cellSize / 2
                    );

                    _cells[position] = cell;
                    clickObservables.Add(cell.OnClick);
                }
            }

            OnCellClicked = clickObservables.Merge();
        }

        public void SpawnPiece(PieceModel pieceModel)
        {
            if (_pieces.ContainsKey(pieceModel.Position)) return;

            var pieceView = Instantiate(_piecePrefab, _boardContainer);
            var position = GetWorldPosition(pieceModel.Position);
            pieceView.Initialize(pieceModel, position);
            _pieces[pieceModel.Position] = pieceView;
        }

        public void MovePiece(Move move)
        {
            if (!_pieces.ContainsKey(move.From))
            {
                Debug.LogWarning($"Не найдено на доске {move.From}");
                return;
            }

            var pieceView = _pieces[move.From];
            _pieces.Remove(move.From);

            BoardPosition[] pathPositions = (move.Path != null && move.Path.Length > 0)
                ? move.Path
                : new BoardPosition[] { move.To };

            var worldPath = pathPositions.Select(GetWorldPosition).ToArray();

            _pieces[move.To] = pieceView;

            UniTask.Void(async () =>
            {
                try
                {
                    await pieceView.MoveAlongPath(worldPath);
                    pieceView.UpdateAppearance();
                }
                catch (Exception ex)
                {
                    Debug.LogError($"Ошибка: {ex}");
                }
            });
        }

        public void RemovePiece(BoardPosition position)
        {
            if (!_pieces.TryGetValue(position, out var pieceView))
                return;

            Destroy(pieceView.gameObject);
            _pieces.Remove(position);
        }

        public void HighlightCells(BoardPosition[] positions)
        {
            foreach (var cell in _cells.Values)
                cell.SetHighlight(false);

            foreach (var position in positions)
                if (_cells.ContainsKey(position))
                    _cells[position].SetHighlight(true);
        }

        public void HighlightLastMove(Move move)
        {
            foreach (var cell in _cells.Values)
                cell.SetLastMove(false);

            if (move == null) return;

            if (_cells.ContainsKey(move.From))
                _cells[move.From].SetLastMove(true);
            if (_cells.ContainsKey(move.To))
                _cells[move.To].SetLastMove(true);
        }

        public void ClearBoard()
        {
            foreach (var pieceView in _pieces.Values)
                Destroy(pieceView.gameObject);

            _pieces.Clear();
        }

        private Vector3 GetWorldPosition(BoardPosition position)
        {
            return new Vector3(
                position.Column * _cellSize - (GameConstants.BoardSize * _cellSize) / 2 + _cellSize / 2,
                0.5f,
                position.Row * _cellSize - (GameConstants.BoardSize * _cellSize) / 2 + _cellSize / 2
            );
        }

        private void OnDestroy()
        {
            ClearBoard();
        }
    }
}