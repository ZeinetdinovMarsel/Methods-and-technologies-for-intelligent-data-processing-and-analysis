using Checkers.Common;
using Checkers.Core;
using Checkers.Services;
using Checkers.Views;
using UnityEngine;
using Zenject;

public class GameInstaller : MonoInstaller
{
    [SerializeField,InjectOptional] private GameSettings _settings;

    public override void InstallBindings()
    {
        Container.Bind<GameModel>().AsSingle().NonLazy();
        Container.Bind<BoardModel>().AsSingle().NonLazy();
        Container.Bind<IGameRulesService>().To<GameRulesService>().AsSingle();

        Container.Bind<IGameService>().To<GameService>().AsSingle()
            .OnInstantiated<GameService>((context, svc) =>
            {
                svc.SetPlayers(_settings.WhitePlayer, _settings.BlackPlayer);
            });

        BindAI(_settings.WhitePlayerAIType, _settings.WhitePlayer, PlayerType.White);
        BindAI(_settings.BlackPlayerAIType, _settings.BlackPlayer, PlayerType.Black);

        Container.Bind<BoardView>().FromComponentInHierarchy().AsSingle();
        Container.Bind<UIView>().FromComponentInHierarchy().AsSingle();
    }

    private void BindAI(AIType aiType, PlayerControlType controlType, PlayerType playerType)
    {
        if (controlType != PlayerControlType.AI) return;

        var binding = Container.Bind<IAIService>().WithId(playerType);

        switch (aiType)
        {
            case AIType.Random:
                binding.To<RandomAIService>().AsCached();
                break;
            case AIType.MinimaxAlphaBeta:
                binding.To<MinimaxAlphaBetaAIService>().AsCached();
                break;
            case AIType.RL:
                binding.To<ReinforcementLearningAIService>().AsCached()
                       .WithArguments(_settings.TcpHost, _settings.TcpPort);
                break;
        }
    }
}