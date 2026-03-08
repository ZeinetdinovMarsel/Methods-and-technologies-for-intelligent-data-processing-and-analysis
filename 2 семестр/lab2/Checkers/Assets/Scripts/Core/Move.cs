using Checkers.Common;

namespace Checkers.Core
{
    public class Move
    {
        public BoardPosition From { get; }
        public BoardPosition To { get; }
        public BoardPosition[] CapturedPieces { get; }
        public BoardPosition[] Path { get; }

        public Move(BoardPosition from, BoardPosition to, BoardPosition[] capturedPieces = null, BoardPosition[] path = null)
        {
            From = from;
            To = to;
            CapturedPieces = capturedPieces ?? new BoardPosition[0];
            Path = path ?? new BoardPosition[0];
        }
    }
}