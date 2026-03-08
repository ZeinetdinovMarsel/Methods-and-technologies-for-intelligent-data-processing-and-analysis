using System.Collections.Generic;
using System.Linq;
using Checkers.Common;
using Checkers.Core;

namespace Checkers.Services
{
    public class GameRulesService : IGameRulesService
    {
        public Move[] GetValidMoves(BoardModel board, PlayerType player)
        {
            var moves = new List<Move>();

            for (int r = 0; r < GameConstants.BoardSize; r++)
            {
                for (int c = 0; c < GameConstants.BoardSize; c++)
                {
                    var piece = board.GetPiece(new BoardPosition(r, c));
                    if (piece == null || piece.Owner != player) continue;

                    moves.AddRange(GetValidMovesForPiece(board, piece));
                }
            }

            var captures = moves.Where(m => m.CapturedPieces.Length > 0).ToArray();
            return captures.Length > 0 ? captures : moves.ToArray();
        }

        public Move[] GetValidMovesForPiece(BoardModel board, PieceModel piece)
        {
            if (piece == null) return new Move[0];

            bool mustCapture = PlayerHasCapture(board, piece.Owner);

            var moves = piece.IsKing
                ? GetKingMoves(board, piece)
                : GetManMoves(board, piece);

            if (mustCapture)
                return moves.Where(m => m.CapturedPieces.Length > 0).ToArray();

            return moves.ToArray();
        }

        private IEnumerable<Move> GetManMoves(BoardModel board, PieceModel piece)
        {
            var directions = GetMoveDirections(piece);

            foreach (var dir in directions)
            {
                var simple = GetSimpleMove(board, piece, dir);
                if (simple != null) yield return simple;

                foreach (var cap in GetManCaptureChains(board, piece, piece.Position, new List<BoardPosition>(), new List<BoardPosition>()))
                    yield return cap;
            }
        }

        private IEnumerable<Move> GetKingMoves(BoardModel board, PieceModel piece)
        {
            var dirs = new[] { (1, 1), (1, -1), (-1, 1), (-1, -1) };

            foreach (var dir in dirs)
            {
                int r = piece.Position.Row + dir.Item1;
                int c = piece.Position.Column + dir.Item2;

                while (board.IsValidPosition(new BoardPosition(r, c)))
                {
                    var pos = new BoardPosition(r, c);

                    if (board.GetPiece(pos) != null)
                        break;

                    if (board.IsBlackCell(pos))
                        yield return new Move(piece.Position, pos, path: new BoardPosition[] { pos });

                    r += dir.Item1;
                    c += dir.Item2;
                }
            }

            foreach (var cap in GetKingCaptureChains(board, piece, piece.Position, new List<BoardPosition>(), new List<BoardPosition>()))
                yield return cap;
        }

        private IEnumerable<Move> GetManCaptureChains(BoardModel board, PieceModel piece, BoardPosition start, List<BoardPosition> captured, List<BoardPosition> path)
        {
            bool found = false;

            foreach (var dir in GetCaptureDirections(piece))
            {
                var mid = new BoardPosition(start.Row + dir.row, start.Column + dir.col);
                var jump = new BoardPosition(start.Row + dir.row * 2, start.Column + dir.col * 2);

                if (!board.IsValidPosition(jump)) continue;
                if (board.GetPiece(jump) != null) continue;

                var midPiece = board.GetPiece(mid);
                if (midPiece == null || midPiece.Owner == piece.Owner || captured.Contains(mid)) continue;

                found = true;
                var newCaptured = new List<BoardPosition>(captured) { mid };
                var newPath = new List<BoardPosition>(path) { jump };

                foreach (var chain in GetManCaptureChains(board, piece, jump, newCaptured, newPath))
                    yield return chain;
            }

            if (!found && captured.Count > 0)
                yield return new Move(piece.Position, start, captured.ToArray(), path.ToArray());
        }

        private IEnumerable<Move> GetKingCaptureChains(BoardModel board, PieceModel piece, BoardPosition start, List<BoardPosition> captured, List<BoardPosition> path)
        {
            bool found = false;
            var dirs = new[] { (1, 1), (1, -1), (-1, 1), (-1, -1) };

            foreach (var dir in dirs)
            {
                int r = start.Row + dir.Item1;
                int c = start.Column + dir.Item2;

                bool enemyFound = false;
                BoardPosition enemyPos = default;

                while (board.IsValidPosition(new BoardPosition(r, c)))
                {
                    var pos = new BoardPosition(r, c);
                    var p = board.GetPiece(pos);

                    if (captured.Contains(pos))
                    {
                        r += dir.Item1;
                        c += dir.Item2;
                        continue;
                    }

                    if (!enemyFound)
                    {
                        if (p != null)
                        {
                            if (p.Owner == piece.Owner)
                                break;

                            enemyFound = true;
                            enemyPos = pos;
                        }
                    }
                    else
                    {
                        if (p != null) break;

                        found = true;
                        var newCaptured = new List<BoardPosition>(captured) { enemyPos };
                        var newPath = new List<BoardPosition>(path) { pos };

                        foreach (var chain in GetKingCaptureChains(board, piece, pos, newCaptured, newPath))
                            yield return chain;
                    }

                    r += dir.Item1;
                    c += dir.Item2;
                }
            }

            if (!found && captured.Count > 0)
                yield return new Move(piece.Position, start, captured.ToArray(), path.ToArray());
        }

        private (int row, int col)[] GetCaptureDirections(PieceModel piece)
        {
            return new[] { (1, 1), (1, -1), (-1, 1), (-1, -1) };
        }

        private (int row, int col)[] GetMoveDirections(PieceModel piece)
        {
            if (piece.Owner == PlayerType.White)
                return new[] { (-1, 1), (-1, -1) };

            return new[] { (1, 1), (1, -1) };
        }

        private Move GetSimpleMove(BoardModel board, PieceModel piece, (int row, int col) dir)
        {
            var pos = new BoardPosition(
                piece.Position.Row + dir.row,
                piece.Position.Column + dir.col);

            if (!board.IsValidPosition(pos)) return null;
            if (board.GetPiece(pos) != null) return null;
            if (!board.IsBlackCell(pos)) return null;

            return new Move(piece.Position, pos, path: new BoardPosition[] { pos });
        }

        private bool PlayerHasCapture(BoardModel board, PlayerType player)
        {
            for (int r = 0; r < GameConstants.BoardSize; r++)
            {
                for (int c = 0; c < GameConstants.BoardSize; c++)
                {
                    var piece = board.GetPiece(new BoardPosition(r, c));
                    if (piece == null || piece.Owner != player) continue;

                    var moves = piece.IsKing
                        ? GetKingCaptureChains(board, piece, piece.Position, new List<BoardPosition>(), new List<BoardPosition>())
                        : GetManCaptureChains(board, piece, piece.Position, new List<BoardPosition>(), new List<BoardPosition>());

                    if (moves.Any())
                        return true;
                }
            }

            return false;
        }

        public bool IsValidMove(BoardModel board, Move move, PlayerType player)
        {
            var piece = board.GetPiece(move.From);
            if (piece == null || piece.Owner != player) return false;

            var moves = GetValidMovesForPiece(board, piece);

            return moves.Any(m => m.From == move.From && m.To == move.To);
        }

        public void ApplyMove(BoardModel board, Move move)
        {
            var piece = board.GetPiece(move.From);
            if (piece == null) return;

            board.RemovePiece(move.From);

            piece.MoveTo(move.To);
            board.SetPiece(move.To, piece);

            foreach (var pos in move.CapturedPieces)
                board.RemovePiece(pos);

            if (piece.Owner == PlayerType.White && move.To.Row == 0)
                piece.PromoteToKing();

            if (piece.Owner == PlayerType.Black && move.To.Row == GameConstants.BoardSize - 1)
                piece.PromoteToKing();
        }

        public bool IsGameOver(BoardModel board, PlayerType player)
        {
            return GetValidMoves(board, player).Length == 0;
        }
    }
}