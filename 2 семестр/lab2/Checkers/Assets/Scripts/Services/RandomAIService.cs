using System.Linq;
using Cysharp.Threading.Tasks;
using Checkers.Common;
using Checkers.Core;

namespace Checkers.Services
{
    public class RandomAIService : IAIService
    {
        private readonly IGameRulesService _rulesService;

        public RandomAIService(IGameRulesService rulesService)
        {
            _rulesService = rulesService;
        }

        public async UniTask<Move> GetMoveAsync(BoardModel board, PlayerType player)
        {
            await UniTask.Delay(1);

            var validMoves = _rulesService.GetValidMoves(board, player);
            if (validMoves.Length == 0) return null;

            var randomIndex = UnityEngine.Random.Range(0, validMoves.Length);
            return validMoves[randomIndex];
        }
    }
}
