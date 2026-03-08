using UniRx;
using TMPro;
using Checkers.Common;
using Checkers.Core;
using UnityEngine;
using UnityEngine.UI;
using System;

namespace Checkers.Views
{
    public class UIView : MonoBehaviour
    {
        [SerializeField] private TMP_Text _turnText;
        [SerializeField] private TMP_Text _whiteCountText;
        [SerializeField] private TMP_Text _blackCountText;
        [SerializeField] private TMP_Text _winnerText;
        [SerializeField] private GameObject _gameOverPanel;
        [SerializeField] private Button _restartButton;
        [SerializeField] private Button _quitButton;

        public IObservable<Unit> OnRestartClicked => _restartButton.OnClickAsObservable();
        public IObservable<Unit> OnQuitClicked => _quitButton.OnClickAsObservable();

        private readonly CompositeDisposable _disposables = new();

        public void Bind(GameModel gameModel)
        {
            gameModel.CurrentTurn
                .Subscribe(turn =>
                {
                    _turnText.text = turn == PlayerType.White ? "Ход: Белые" : "Ход: Черные";
                })
                .AddTo(_disposables);

            gameModel.WhitePiecesCount
                .Subscribe(count => _whiteCountText.text = $"Белые: {count}")
                .AddTo(_disposables);

            gameModel.BlackPiecesCount
                .Subscribe(count => _blackCountText.text = $"Черные: {count}")
                .AddTo(_disposables);

            gameModel.State
                .Subscribe(state =>
                {
                    _gameOverPanel.SetActive(state == GameState.GameOver);
                })
                .AddTo(_disposables);

            gameModel.Winner
                .Subscribe(winner =>
                {
                    if (winner != PlayerType.None)
                    {
                        _winnerText.text = winner == PlayerType.White ? "Белые победили!" : "Черные победили!";
                    }
                })
                .AddTo(_disposables);
        }

        public void ShowGameOver(PlayerType winner)
        {
            _winnerText.text = winner == PlayerType.White ? "Белые победили!" : "Черные победили!";
            _gameOverPanel.SetActive(true);
        }

        private void OnDestroy()
        {
            _disposables?.Dispose();
        }
    }
}