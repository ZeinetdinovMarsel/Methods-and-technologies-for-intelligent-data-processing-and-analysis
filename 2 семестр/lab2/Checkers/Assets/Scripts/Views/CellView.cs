using Checkers.Core;
using System;
using UniRx;
using UnityEngine;
using UnityEngine.EventSystems;

namespace Checkers.Views
{
    public class CellView : MonoBehaviour, IPointerClickHandler
    {
        [SerializeField] private Renderer _cellRenderer;
        [SerializeField] private Color _whiteCellColor = Color.white;
        [SerializeField] private Color _blackCellColor = Color.gray;
        [SerializeField] private Color _highlightColor = Color.yellow;
        [SerializeField] private Color _lastMoveColor = Color.cyan;

        private readonly Subject<BoardPosition> _clickSubject = new();
        private BoardPosition _position;
        private bool _isHighlighted;
        private bool _isLastMove;

        public IObservable<BoardPosition> OnClick => _clickSubject;

        public void Initialize(BoardPosition position, bool isBlackCell)
        {
            _position = position;
            _cellRenderer.material.color = isBlackCell ? _blackCellColor : _whiteCellColor;
        }

        public void OnPointerClick(PointerEventData eventData)
        {
            _clickSubject.OnNext(_position);
        }

        public void SetHighlight(bool isHighlighted)
        {
            _isHighlighted = isHighlighted;
             UpdateColor();
        }

        public void SetLastMove(bool isLastMove)
        {
            _isLastMove = isLastMove;
            UpdateColor();
        }

        private void UpdateColor()
        {
            if (_isLastMove && !_isHighlighted)
                _cellRenderer.material.color = _lastMoveColor;
            else if (_isHighlighted)
                _cellRenderer.material.color = _highlightColor;
            else
                _cellRenderer.material.color = ((int)(_position.Row + _position.Column)) % 2 == 1
                    ? _blackCellColor
                    : _whiteCellColor;
        }

        private void OnDestroy()
        {
            _clickSubject?.OnCompleted();
            _clickSubject?.Dispose();
        }
    }
}