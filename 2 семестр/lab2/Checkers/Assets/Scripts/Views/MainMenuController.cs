using Checkers.Common;
using TMPro;
using UnityEngine;
using UnityEngine.SceneManagement;
using UnityEngine.UI;
using Zenject;

public class MainMenuController : MonoBehaviour
{
    [SerializeField] private TMP_InputField _hostInput;
    [SerializeField] private TMP_InputField _portInput;
    [SerializeField] private TMP_Dropdown _whiteType;
    [SerializeField] private TMP_Dropdown _blackType;
    [SerializeField] private TMP_Dropdown _whiteAIType;
    [SerializeField] private TMP_Dropdown _blackAIType;
    public void OnStartGameClicked()
    {
        var settings = new GameSettings
        {
            TcpHost = _hostInput.text,
            TcpPort = int.Parse(_portInput.text),
            WhitePlayer = (PlayerControlType)_whiteType.value,
            BlackPlayer = (PlayerControlType)_blackType.value,
            WhitePlayerAIType = (AIType)_whiteAIType.value,
            BlackPlayerAIType = (AIType)_blackAIType.value
        };

        ProjectContext.Instance.Container.Bind<GameSettings>().FromInstance(settings).AsSingle();

        SceneManager.LoadScene("GameScene");
    }
}