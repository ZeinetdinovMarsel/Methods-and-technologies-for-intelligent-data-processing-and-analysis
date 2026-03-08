using ArtificeToolkit.Attributes;
using Checkers.Common;
using Checkers.Core;
using Checkers.Services;
using Checkers.Views;
using UnityEngine;
using Zenject;

namespace Checkers.Installers
{
    public class GameInstaller : MonoInstaller
    {
        [SerializeField] private string _tcpHost;
        [SerializeField] private int _tcpPort;

        [SerializeField] private PlayerControlType WhitePlayer = PlayerControlType.Human;
        [EnableIf(nameof(IsWhitePlayerAI)), SerializeField]
        private AIType WhitePlayerAIType = AIType.Random;

        [SerializeField] private PlayerControlType BlackPlayer = PlayerControlType.AI;
        [EnableIf(nameof(IsBlackPlayerAI)), SerializeField] 
        private AIType BlackPlayerAIType = AIType.Random;

        public override void InstallBindings()
        {
            Container.Bind<GameModel>().FromNew().AsSingle().NonLazy();
            Container.Bind<BoardModel>().FromNew().AsSingle().NonLazy();

            Container.Bind<IGameRulesService>().To<GameRulesService>().FromNew().AsSingle();


            Container.Bind<IGameService>().To<GameService>().FromNew().AsSingle()
                .OnInstantiated<GameService>((context, svc) =>
                {
                    svc.SetPlayers(WhitePlayer, BlackPlayer);
                });

            SetAI(ref WhitePlayerAIType, ref WhitePlayer, PlayerType.White);
            SetAI(ref BlackPlayerAIType, ref BlackPlayer, PlayerType.Black);

            Container.Bind<BoardView>().FromComponentInHierarchy().AsSingle();
            Container.Bind<UIView>().FromComponentInHierarchy().AsSingle();
        }

        private bool IsWhitePlayerAI() => WhitePlayer == PlayerControlType.AI;
        private bool IsBlackPlayerAI() => BlackPlayer == PlayerControlType.AI;

        private void SetAI(ref AIType AIType, ref PlayerControlType playerControl, PlayerType playerType)
        {
            if (playerControl != PlayerControlType.AI) { return; }

            switch (AIType)
            {
                case AIType.Random:
                    Container.Bind<IAIService>()
                        .WithId(playerType)
                        .To<RandomAIService>()
                        .FromNew()
                        .AsCached();
                    break;
                case AIType.MinimaxAlphaBeta:
                    Container.Bind<IAIService>()
                        .WithId(playerType)
                        .To<MinimaxAlphaBetaAIService>()
                        .FromNew()
                        .AsCached();
                    break;
                case AIType.RL:
                    Container.Bind<IAIService>()
                        .WithId(playerType)
                        .To<ReinforcementLearningAIService>()
                        .FromNew()
                        .AsCached()
                        .WithArguments(_tcpHost, _tcpPort);
                    break;
            }

        }
    }
}