using Checkers.Common;
[System.Serializable]
public class GameSettings
{
    public string TcpHost = "127.0.0.1";
    public int TcpPort = 5555;
    public PlayerControlType WhitePlayer = PlayerControlType.Human;
    public AIType WhitePlayerAIType = AIType.MinimaxAlphaBeta;
    public PlayerControlType BlackPlayer = PlayerControlType.Human;
    public AIType BlackPlayerAIType = AIType.MinimaxAlphaBeta;
}