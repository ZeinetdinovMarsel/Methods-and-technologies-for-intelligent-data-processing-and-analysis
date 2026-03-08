using Cysharp.Threading.Tasks;
using Checkers.Common;
using Checkers.Core;

namespace Checkers.Services
{
    public interface IAIService
    {
        UniTask<Move> GetMoveAsync(BoardModel board, PlayerType player);
    }
}
