using Checkers.Common;

namespace Checkers.Core
{
    public class PieceModel
    {
        public PlayerType Owner { get; private set; }
        public bool IsKing { get; private set; }
        public BoardPosition Position { get; private set; }

        public PieceModel(PlayerType owner, BoardPosition position)
        {
            Owner = owner;
            Position = position;
            IsKing = false;
        }
        public PieceModel Clone()
        {
            var clone = new PieceModel(this.Owner, this.Position);
            if (this.IsKing)
                clone.PromoteToKing();
            return clone;
        }
        public void PromoteToKing() => IsKing = true;
        public void DemoteFromKing() => IsKing = false;
        public void MoveTo(BoardPosition newPosition) => Position = newPosition;


    }
}