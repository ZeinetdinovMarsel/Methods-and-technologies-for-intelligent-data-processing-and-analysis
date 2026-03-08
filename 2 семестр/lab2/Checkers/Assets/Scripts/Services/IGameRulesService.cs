using Checkers.Common;
using Checkers.Core;
using System.Collections.Generic;

namespace Checkers.Services
{
    public interface IGameRulesService
    {
        Move[] GetValidMoves(BoardModel board, PlayerType player);
        Move[] GetValidMovesForPiece(BoardModel board, PieceModel piece);
        bool IsValidMove(BoardModel board, Move move, PlayerType player);
        void ApplyMove(BoardModel board, Move move);
        bool IsGameOver(BoardModel board, PlayerType currentPlayer);
    }
}