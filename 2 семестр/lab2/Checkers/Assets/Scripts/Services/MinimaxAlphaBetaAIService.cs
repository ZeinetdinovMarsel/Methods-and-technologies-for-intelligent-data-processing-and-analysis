using Checkers.Common;
using Checkers.Core;
using Cysharp.Threading.Tasks;
using System.Collections.Generic;
using Unity.VisualScripting;
using UnityEngine;

namespace Checkers.Services
{
    public class MinimaxAlphaBetaAIService : IAIService
    {
        private readonly IGameRulesService _rules;
        private const int MAX_DEPTH = 6;

        private BoardModel _aiBoard;

        public MinimaxAlphaBetaAIService(IGameRulesService rules)
        {
            _rules = rules;
        }

        public async UniTask<Move> GetMoveAsync(BoardModel board, PlayerType player)
        {
            return await UniTask.RunOnThreadPool(() =>
            {
                PrepareAIBoard(board);

                Move bestMove = null;
                float bestScore = float.NegativeInfinity;

                var moves = _rules.GetValidMoves(_aiBoard, player);
                var sortedMoves = new List<Move>(moves);
                sortedMoves.Sort((a, b) => b.CapturedPieces.Length.CompareTo(a.CapturedPieces.Length));

                foreach (var move in sortedMoves)
                {
                    var undo = ApplyMove(_aiBoard, move);
                    if (undo.MovingPiece == null) continue;

                    float score = Minimax(_aiBoard, MAX_DEPTH - 1, false, player, float.NegativeInfinity, float.PositiveInfinity);
                    UndoMove(_aiBoard, undo);

                    if (score > bestScore)
                    {
                        bestScore = score;
                        bestMove = move;
                    }
                }

                return bestMove;
            });
        }

        private void PrepareAIBoard(BoardModel liveBoard)
        {
            if (_aiBoard == null)
                _aiBoard = new BoardModel();

            _aiBoard.Clear();

            foreach (var piece in liveBoard.GetAllPieces())
            {
                _aiBoard.SetPiece(piece.Position, piece.Clone());
            }
        }

        private float Minimax(BoardModel board, int depth, bool maximizing, PlayerType aiPlayer, float alpha, float beta)
        {
            PlayerType player = maximizing ? aiPlayer : Opponent(aiPlayer);

            if (depth == 0 || _rules.IsGameOver(board, player))
                return Evaluate(board, aiPlayer);

            var moves = _rules.GetValidMoves(board, player);
            var sortedMoves = new List<Move>(moves);
            sortedMoves.Sort((a, b) => b.CapturedPieces.Length.CompareTo(a.CapturedPieces.Length));

            if (maximizing)
            {
                float best = float.NegativeInfinity;
                foreach (var move in sortedMoves)
                {
                    var undo = ApplyMove(board, move);
                    if (undo.MovingPiece == null) continue;

                    float score = Minimax(board, depth - 1, false, aiPlayer, alpha, beta);
                    UndoMove(board, undo);

                    best = Mathf.Max(best, score);
                    alpha = Mathf.Max(alpha, score);
                    if (beta <= alpha) break;
                }
                return best;
            }
            else
            {
                float best = float.PositiveInfinity;
                foreach (var move in sortedMoves)
                {
                    var undo = ApplyMove(board, move);
                    if (undo.MovingPiece == null) continue;

                    float score = Minimax(board, depth - 1, true, aiPlayer, alpha, beta);
                    UndoMove(board, undo);

                    best = Mathf.Min(best, score);
                    beta = Mathf.Min(beta, score);
                    if (beta <= alpha) break;
                }
                return best;
            }
        }

        private float Evaluate(BoardModel board, PlayerType ai)
        {
            float score = 0;

            foreach (var piece in board.GetAllPieces())
            {
                float value = piece.IsKing ? 6f : 1f;

                if (!piece.IsKing)
                {
                    value += piece.Owner == PlayerType.White
                        ? (7 - piece.Position.Row) * 0.1f
                        : piece.Position.Row * 0.1f;
                }

                score += piece.Owner == ai ? value : -value;
            }

            return score;
        }

        private PlayerType Opponent(PlayerType p) => p == PlayerType.White ? PlayerType.Black : PlayerType.White;

        private UndoData ApplyMove(BoardModel board, Move move)
        {
            var movingPiece = board.GetPiece(move.From);
            if (movingPiece == null) return default;

            PieceModel[] captured = new PieceModel[move.CapturedPieces.Length];
            for (int i = 0; i < move.CapturedPieces.Length; i++)
                captured[i] = board.GetPiece(move.CapturedPieces[i]);

            bool wasKing = movingPiece.IsKing;
            _rules.ApplyMove(board, move);

            return new UndoData
            {
                Move = move,
                MovingPiece = movingPiece,
                Captured = captured,
                WasKing = wasKing
            };
        }

        private void UndoMove(BoardModel board, UndoData undo)
        {
            if (undo.MovingPiece == null) return;

            var move = undo.Move;
            var piece = undo.MovingPiece;

            board.SetPiece(move.To, null);
            piece.MoveTo(move.From);

            if (undo.WasKing && !piece.IsKing)
                piece.PromoteToKing();
            else if (!undo.WasKing && piece.IsKing)
                piece.DemoteFromKing();

            board.SetPiece(move.From, piece);

            for (int i = 0; i < undo.Captured.Length; i++)
            {
                var captured = undo.Captured[i];
                if (captured == null) continue;

                var pos = move.CapturedPieces[i];
                captured.MoveTo(pos);
                board.SetPiece(pos, captured);
            }
        }

        struct UndoData
        {
            public Move Move;
            public PieceModel MovingPiece;
            public PieceModel[] Captured;
            public bool WasKing;
        }
    }

    public static class BoardModelExtensions
    {
        public static IEnumerable<PieceModel> GetAllPieces(this BoardModel board)
        {
            for (int r = 0; r < GameConstants.BoardSize; r++)
            {
                for (int c = 0; c < GameConstants.BoardSize; c++)
                {
                    var piece = board.GetPiece(new BoardPosition(r, c));
                    if (piece != null) yield return piece;
                }
            }
        }
    }
}