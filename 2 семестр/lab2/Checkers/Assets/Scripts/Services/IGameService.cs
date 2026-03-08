using Checkers.Core;
using Cysharp.Threading.Tasks;
using System.Threading;

namespace Checkers.Services
{
    public interface IGameService
    {
        void SetPlayers(PlayerControlType white, PlayerControlType black);
        PlayerControlType GetCurrentPlayerControl();
        UniTask HandleTurnAsync();
        void Initialize();
        void MakeMove(Move move);
        UniTask<Move> GetAIMoveAsync();
        void ResetGame();
    }
}
