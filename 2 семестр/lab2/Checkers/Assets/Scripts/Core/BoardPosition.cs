using System;

namespace Checkers.Core
{
    [Serializable]
    public struct BoardPosition : IEquatable<BoardPosition>
    {
        public int Row { get; }
        public int Column { get; }

        public BoardPosition(int row, int column)
        {
            Row = row;
            Column = column;
        }

        public bool Equals(BoardPosition other) => Row == other.Row && Column == other.Column;
        public override bool Equals(object obj) => obj is BoardPosition other && Equals(other);
        public override int GetHashCode() => HashCode.Combine(Row, Column);
        public override string ToString() => $"({Row}, {Column})";

        public static bool operator ==(BoardPosition left, BoardPosition right) => left.Equals(right);
        public static bool operator !=(BoardPosition left, BoardPosition right) => !left.Equals(right);
    }
}