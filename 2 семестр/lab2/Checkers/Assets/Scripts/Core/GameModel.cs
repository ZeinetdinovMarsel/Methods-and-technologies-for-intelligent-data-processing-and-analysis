using UniRx;
using Checkers.Common;

namespace Checkers.Core
{
    public class GameModel
    {
        public ReactiveProperty<GameState> State { get; } = new(GameState.NotStarted);
        public ReactiveProperty<PlayerType> CurrentTurn { get; } = new(PlayerType.White);
        public ReactiveProperty<PlayerType> Winner { get; } = new(PlayerType.None);
        public ReactiveProperty<int> WhitePiecesCount { get; } = new(12);
        public ReactiveProperty<int> BlackPiecesCount { get; } = new(12);
        public ReactiveProperty<Move> LastMove { get; } = new(null);
    }
}