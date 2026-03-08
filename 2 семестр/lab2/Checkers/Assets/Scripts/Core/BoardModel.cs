using Checkers.Common;

namespace Checkers.Core
{
    public class BoardModel
    {
        private readonly PieceModel[,] _pieces;

        public BoardModel()
        {
            _pieces = new PieceModel[GameConstants.BoardSize, GameConstants.BoardSize];
        }
        public PieceModel GetPiece(BoardPosition position)
        {
            if (!IsValidPosition(position)) return null;
            return _pieces[position.Row, position.Column];
        }

        public void SetPiece(BoardPosition position, PieceModel piece)
        {
            if (!IsValidPosition(position)) return;
            _pieces[position.Row, position.Column] = piece;
        }

        public void RemovePiece(BoardPosition position)
        {
            if (!IsValidPosition(position)) return;
            _pieces[position.Row, position.Column] = null;
        }

        public bool IsValidPosition(BoardPosition position)
        {
            return position.Row >= 0 && position.Row < GameConstants.BoardSize &&
                   position.Column >= 0 && position.Column < GameConstants.BoardSize;
        }

        public bool IsBlackCell(BoardPosition position)
        {
            return (position.Row + position.Column) % 2 == 0;
        }
        public void Clear()
        {
            for (int r = 0; r < GameConstants.BoardSize; r++)
                for (int c = 0; c < GameConstants.BoardSize; c++)
                    _pieces[r, c] = null;
        }
    }
}