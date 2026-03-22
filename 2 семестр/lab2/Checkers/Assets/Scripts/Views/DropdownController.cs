using TMPro;
using UnityEngine;

public class DropdownController : MonoBehaviour
{
    [SerializeField] private int _targetValue=0;
    [SerializeField] private Transform _objectToEnable;

    [SerializeField] private TMP_Dropdown _dropdown;

    private void Start()
    {
        _dropdown = GetComponentInChildren<TMP_Dropdown>();
        _dropdown.onValueChanged.AddListener((value) =>
        {
            _objectToEnable.gameObject.SetActive(value == _targetValue);
        });
    }
}
