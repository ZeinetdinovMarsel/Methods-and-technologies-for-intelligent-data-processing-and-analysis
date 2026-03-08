using Checkers.Common;
using Checkers.Core;
using Cysharp.Threading.Tasks;
using PrimeTween;
using System;
using System.Threading;
using UnityEngine;

namespace Checkers.Views
{
    public class PieceView : MonoBehaviour
    {
        [SerializeField] private Renderer _pieceRenderer;
        [SerializeField] private Color _whiteColor = Color.white;
        [SerializeField] private Color _blackColor = Color.black;
        [SerializeField] private Color _kingColor = Color.yellow;
        [SerializeField] private Transform _kingIndicator;

        private PieceModel _pieceModel;
        private Vector3 _targetPosition;

        private CancellationTokenSource _cts;

        public void Initialize(PieceModel pieceModel, Vector3 position)
        {
            _pieceModel = pieceModel;
            _targetPosition = position;
            transform.position = position;
            UpdateAppearance();
        }

        public async UniTask MoveAlongPath(Vector3[] worldPath)
        {
            _cts?.Cancel();
            _cts = new CancellationTokenSource();
            CancellationToken token = _cts.Token;

            try
            {
                foreach (var target in worldPath)
                {
                    token.ThrowIfCancellationRequested();
                    await MoveToPositionSmooth(target, 0.01f, token);
                }
            }
            catch (OperationCanceledException)
            {
            }
        }
        public async UniTask MoveToPositionSmooth(Vector3 target, float duration, CancellationToken token)
        {
            Vector3 start = transform.position;
            float elapsed = 0f;

            while (elapsed < duration)
            {
                token.ThrowIfCancellationRequested();

                elapsed += Time.deltaTime;
                transform.position = Vector3.Lerp(start, target, elapsed / duration);
                await UniTask.Yield(token);
            }

            transform.position = target;
        }

        public void MoveTo(Vector3 newPosition)
        {
            _targetPosition = newPosition;

            UniTask.Void(async () =>
            {
                await MoveAlongPath(new Vector3[] { newPosition });
            });
        }

        public void UpdateAppearance()
        {
            if (_pieceModel == null) return;
            if (_pieceRenderer != null)
            {
                if (_pieceRenderer.gameObject != null)
                {
                    _pieceRenderer.material.color = _pieceModel.Owner == PlayerType.White
                        ? _whiteColor
                        : _blackColor;
                }
            }

            if (_kingIndicator != null && _kingIndicator.gameObject != null)
            {
                _kingIndicator.gameObject.SetActive(_pieceModel.IsKing);
            }
        }

        public void Remove()
        {
            _cts?.Cancel();
            _cts = null;

            Tween.Scale(transform, Vector3.zero, 0.2f, Ease.InQuad)
                .OnComplete(() => Destroy(gameObject));
        }

        private void OnDestroy()
        {
            _cts?.Cancel();
            _cts = null;
        }
    }
}