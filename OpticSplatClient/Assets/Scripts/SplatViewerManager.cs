// VERSION: 2.6.1 - MINIMALIST HTML STYLE UI
using UnityEngine;
using UnityEngine.UI;
using GaussianSplatting.Runtime;
using System.IO;
using UnityEngine.InputSystem;
using System.Collections;
using UnityEngine.Networking;
using System.Collections.Generic;
using System.Diagnostics;
using System.Text;

#if UNITY_EDITOR
using UnityEditor;
using GaussianSplatting.Editor;
#endif

[ExecuteAlways]
public class SplatViewerManager : MonoBehaviour
{
    [System.Serializable]
    public class GaussianMetadata
    {
        public string id;
        public string type;
        public float focal_length_35mm;
        public int width;
        public int height;
        public float aspect_ratio;
        public string filename;
    }

    [System.Serializable]
    public class GenerateResponse 
    { 
        public string id; 
        public string ply_url;
        public GaussianMetadata metadata;
    }

    [Header("Core References")]
    public GaussianSplatRenderer splatRenderer;
    public OrbitCameraController cameraController;
    public Camera mainCamera; 
    public Button loadButton;
    public Button generateButton;
    public Text statusText;
    public Text debugConsoleText;

    [Header("Current State")]
    public GaussianMetadata currentMetadata;

    [Header("Generated UI (DO NOT MANUAL ASSIGN)")]
    public Slider focalSlider;
    public InputField focalInput;
    public Slider panSlider;
    public InputField panInput;
    public Slider tiltSlider;
    public InputField tiltInput;
    public Slider rollSlider;
    public InputField rollInput;
    public Slider distSlider;
    public InputField distInput;
    public Button resetButton;
    public Button captureButton;
    public Button[] aspectButtons; 

    public bool triggerUICreation = false;

    private const string BackendUrl = "http://127.0.0.1:8000";
    private const string PythonExePath = @"C:\git\OpticSplat\backend\build_env\Scripts\python.exe";
    private const string ServerScriptPath = @"C:\git\OpticSplat\backend\server.py";
    private const string BackendExePath = @"C:\git\OpticSplat\backend\dist\server.exe";
    
    private List<string> consoleLines = new List<string>();
    private Process backendProcess;
    private readonly object logLock = new object();
    private bool isGenerating = false;
    private bool isUpdatingUI = false;

    // HTML Color Palette
    private readonly Color kColorBg = new Color(0.1f, 0.1f, 0.1f, 0.98f);
    private readonly Color kColorAccent = new Color(0.23f, 0.51f, 0.96f); // Blue 500
    private readonly Color kColorTextDim = new Color(1, 1, 1, 0.4f);
    private readonly Color kColorBorder = new Color(1, 1, 1, 0.1f);
    private readonly Color kColorInputBg = new Color(1, 1, 1, 0.05f);

    void Start()
    {
        if (Application.isPlaying)
        {
            if (mainCamera == null) mainCamera = Camera.main;

            if (loadButton != null)
                loadButton.onClick.AddListener(OnLoadButtonClicked);
            
            if (generateButton != null)
                generateButton.onClick.AddListener(OnGenerateButtonClicked);

            FindAndLinkUI();
            
            LogToConsole("Viewer initialized.");
            UpdateStatusText();
        }
    }

    void CreateSidebarUI()
    {
        var canvasGo = GameObject.Find("UI Canvas");
        Canvas canvas = canvasGo != null ? canvasGo.GetComponent<Canvas>() : Object.FindFirstObjectByType<Canvas>();
        if (canvas == null) 
        {
            LogToConsole("ERROR: Could not find UI Canvas.");
            return;
        }

        var existing = GameObject.Find("SidebarContainer");
        if (existing != null) DestroyImmediate(existing);

        // 1. Sidebar Main Container
        var containerGo = new GameObject("SidebarContainer", typeof(RectTransform), typeof(Image), typeof(ScrollRect));
        containerGo.transform.SetParent(canvas.transform, false);
        var contRT = containerGo.GetComponent<RectTransform>();
        contRT.anchorMin = new Vector2(1, 0); contRT.anchorMax = new Vector2(1, 1);
        contRT.pivot = new Vector2(1, 0.5f);
        contRT.offsetMin = new Vector2(-300, 0); contRT.offsetMax = new Vector2(0, 0);
        containerGo.GetComponent<Image>().color = kColorBg;

        // 2. Scroll Viewport
        var viewportGo = new GameObject("Viewport", typeof(RectTransform), typeof(Mask), typeof(Image));
        viewportGo.transform.SetParent(containerGo.transform, false);
        viewportGo.GetComponent<Image>().color = new Color(0,0,0,0);
        var viewRT = viewportGo.GetComponent<RectTransform>();
        viewRT.anchorMin = Vector2.zero; viewRT.anchorMax = Vector2.one; 
        viewRT.offsetMin = Vector2.zero; viewRT.offsetMax = Vector2.zero;

        // 3. Content Area
        var sidebarGo = new GameObject("SidebarContent", typeof(RectTransform), typeof(VerticalLayoutGroup), typeof(ContentSizeFitter));
        sidebarGo.transform.SetParent(viewportGo.transform, false);
        var sbRT = sidebarGo.GetComponent<RectTransform>();
        sbRT.anchorMin = new Vector2(0, 1); sbRT.anchorMax = new Vector2(1, 1);
        sbRT.pivot = new Vector2(0.5f, 1); sbRT.sizeDelta = Vector2.zero;
        
        var vlg = sidebarGo.GetComponent<VerticalLayoutGroup>();
        vlg.padding = new RectOffset(20, 20, 30, 30);
        vlg.spacing = 18;
        vlg.childControlHeight = true; vlg.childForceExpandHeight = false;
        vlg.childControlWidth = true; vlg.childForceExpandWidth = true;
        sidebarGo.GetComponent<ContentSizeFitter>().verticalFit = ContentSizeFitter.FitMode.PreferredSize;

        var scroll = containerGo.GetComponent<ScrollRect>();
        scroll.content = sbRT; scroll.viewport = viewRT;
        scroll.horizontal = false; scroll.vertical = true;

        // --- SECTION: OPTICS ---
        AddSectionHeader(sidebarGo.transform, "Optics");
        AddSlider(sidebarGo.transform, "Focal Length", 8, 400, 35, "SLIDER_FOCAL", "mm");

        // --- SECTION: ORIENTATION ---
        AddSectionHeader(sidebarGo.transform, "Orientation");
        AddSlider(sidebarGo.transform, "Pan", -180, 180, 0, "SLIDER_PAN", "°");
        AddSlider(sidebarGo.transform, "Tilt", -80, 80, 0, "SLIDER_TILT", "°");
        AddSlider(sidebarGo.transform, "Roll", -180, 180, 0, "SLIDER_ROLL", "°");

        // --- SECTION: POSITION ---
        AddSectionHeader(sidebarGo.transform, "Position");
        AddSlider(sidebarGo.transform, "Distance", 0.1f, 100, 5, "SLIDER_DIST", "m");
        
        // --- SECTION: ACTIONS ---
        AddSectionHeader(sidebarGo.transform, "Actions");
        var actionsRow = new GameObject("ActionsRow", typeof(RectTransform), typeof(HorizontalLayoutGroup), typeof(LayoutElement));
        actionsRow.transform.SetParent(sidebarGo.transform, false);
        var alg = actionsRow.GetComponent<HorizontalLayoutGroup>();
        alg.spacing = 8; alg.childControlWidth = true; alg.childForceExpandWidth = true;
        alg.childControlHeight = true; alg.childForceExpandHeight = false;
        actionsRow.GetComponent<LayoutElement>().preferredHeight = 30;
        CreateButton(actionsRow.transform, "Reset View", "BTN_RESET");
        CreateButton(actionsRow.transform, "Capture View", "BTN_CAPTURE");

        // --- SECTION: ASPECT RATIO ---
        AddSectionHeader(sidebarGo.transform, "Aspect Ratio");
        var aspectRow = new GameObject("AspectRow", typeof(RectTransform), typeof(GridLayoutGroup), typeof(LayoutElement));
        aspectRow.transform.SetParent(sidebarGo.transform, false);
        var glg = aspectRow.GetComponent<GridLayoutGroup>();
        glg.cellSize = new Vector2(48, 28); glg.spacing = new Vector2(6, 6);
        aspectRow.GetComponent<LayoutElement>().preferredHeight = 64;
        
        string[] labels = {"ORG", "16:9", "4:3", "1:1", "9:16"};
        for(int i=0; i<labels.Length; i++)
        {
            CreateButton(aspectRow.transform, labels[i], "BTN_ASPECT_" + labels[i]);
        }

        FindAndLinkUI();

#if UNITY_EDITOR
        EditorUtility.SetDirty(this);
#endif
    }

    void AddSectionHeader(Transform parent, string title)
    {
        var go = new GameObject("Header_" + title, typeof(RectTransform), typeof(Text), typeof(LayoutElement));
        go.transform.SetParent(parent, false);
        var t = go.GetComponent<Text>();
        t.text = title.ToUpper();
        t.font = Resources.GetBuiltinResource<Font>("LegacyRuntime.ttf");
        t.fontSize = 11; t.fontStyle = FontStyle.Bold;
        t.color = kColorTextDim; t.alignment = TextAnchor.MiddleLeft;
        go.GetComponent<LayoutElement>().preferredHeight = 20;
    }

    void AddSlider(Transform parent, string label, float min, float max, float val, string name, string unit = "")
    {
        var root = new GameObject(name + "_ROOT", typeof(RectTransform), typeof(VerticalLayoutGroup), typeof(LayoutElement));
        root.transform.SetParent(parent, false);
        var vlg = root.GetComponent<VerticalLayoutGroup>();
        vlg.childControlHeight = true; vlg.childForceExpandHeight = false; vlg.spacing = 4;
        root.GetComponent<LayoutElement>().preferredHeight = 44;

        // Label Row
        var head = new GameObject("Header", typeof(RectTransform), typeof(HorizontalLayoutGroup));
        head.transform.SetParent(root.transform, false);
        var hlg = head.GetComponent<HorizontalLayoutGroup>();
        hlg.childControlWidth = true; hlg.childForceExpandWidth = false;

        var lGo = new GameObject("Label", typeof(RectTransform), typeof(Text)).GetComponent<Text>();
        lGo.transform.SetParent(head.transform, false);
        lGo.text = label; lGo.font = Resources.GetBuiltinResource<Font>("LegacyRuntime.ttf");
        lGo.fontSize = 12; lGo.color = new Color(0.85f, 0.85f, 0.85f);

        var valRow = new GameObject("ValueRow", typeof(RectTransform), typeof(HorizontalLayoutGroup));
        valRow.transform.SetParent(head.transform, false);
        var vhlg = valRow.GetComponent<HorizontalLayoutGroup>();
        vhlg.childControlWidth = true; vhlg.childForceExpandWidth = false; vhlg.spacing = 2;

        var iGo = new GameObject("Input", typeof(RectTransform), typeof(Image), typeof(InputField));
        iGo.transform.SetParent(valRow.transform, false);
        iGo.GetComponent<Image>().color = kColorInputBg;
        var input = iGo.GetComponent<InputField>();
        var itGo = new GameObject("Text", typeof(RectTransform), typeof(Text)).GetComponent<Text>();
        itGo.transform.SetParent(iGo.transform, false);
        itGo.font = Resources.GetBuiltinResource<Font>("LegacyRuntime.ttf");
        itGo.fontSize = 11; itGo.alignment = TextAnchor.MiddleRight; itGo.color = Color.white;
        input.textComponent = itGo;
        var iRT = iGo.GetComponent<RectTransform>();
        iRT.sizeDelta = new Vector2(38, 18);

        if (!string.IsNullOrEmpty(unit))
        {
            var uGo = new GameObject("Unit", typeof(RectTransform), typeof(Text)).GetComponent<Text>();
            uGo.transform.SetParent(valRow.transform, false);
            uGo.text = unit; uGo.font = Resources.GetBuiltinResource<Font>("LegacyRuntime.ttf");
            uGo.fontSize = 10; uGo.color = kColorTextDim; uGo.alignment = TextAnchor.MiddleLeft;
            uGo.GetComponent<RectTransform>().sizeDelta = new Vector2(15, 18);
        }

        // Slider (No Fill Bar, only Track + Handle)
        var sGo = new GameObject(name, typeof(RectTransform), typeof(Slider));
        sGo.transform.SetParent(root.transform, false);
        var s = sGo.GetComponent<Slider>();
        s.transition = Selectable.Transition.None;
        s.navigation = new Navigation { mode = Navigation.Mode.None };
        sGo.GetComponent<RectTransform>().sizeDelta = new Vector2(0, 16);

        var bg = new GameObject("Background", typeof(RectTransform), typeof(Image));
        bg.transform.SetParent(sGo.transform, false);
        var bgImg = bg.GetComponent<Image>();
        bgImg.color = new Color(1, 1, 1, 0.08f);
        bgImg.raycastTarget = false;
        var bgRt = bg.GetComponent<RectTransform>();
        bgRt.anchorMin = new Vector2(0, 0.48f); bgRt.anchorMax = new Vector2(1, 0.52f); 
        bgRt.sizeDelta = Vector2.zero;
        
        var handleArea = new GameObject("HandleArea", typeof(RectTransform));
        handleArea.transform.SetParent(sGo.transform, false);
        var haRt = handleArea.GetComponent<RectTransform>();
        haRt.anchorMin = Vector2.zero; haRt.anchorMax = Vector2.one; haRt.sizeDelta = new Vector2(-10, 0);

        var handle = new GameObject("Handle", typeof(RectTransform), typeof(Image));
        handle.transform.SetParent(handleArea.transform, false);
        var hImg = handle.GetComponent<Image>();
        hImg.color = Color.white;
        hImg.raycastTarget = false;
        var hRt = handle.GetComponent<RectTransform>();
        hRt.sizeDelta = new Vector2(2, 12);
        s.handleRect = hRt;

        s.minValue = min; s.maxValue = max; s.value = val;
    }

    void CreateButton(Transform parent, string label, string name)
    {
        var go = new GameObject(name, typeof(RectTransform), typeof(Image), typeof(Button), typeof(LayoutElement));
        go.transform.SetParent(parent, false);
        go.GetComponent<LayoutElement>().preferredHeight = 28;
        
        var img = go.GetComponent<Image>();
        img.color = kColorInputBg;
        
        var b = go.GetComponent<Button>();
        b.transition = Selectable.Transition.ColorTint;
        b.navigation = new Navigation { mode = Navigation.Mode.None };
        var cb = b.colors;
        cb.normalColor = kColorInputBg;
        cb.highlightedColor = new Color(1, 1, 1, 0.12f);
        cb.pressedColor = new Color(1, 1, 1, 0.2f);
        cb.selectedColor = kColorInputBg; // Prevent blue selection block
        b.colors = cb;
        
        var t = new GameObject("Text", typeof(RectTransform), typeof(Text)).GetComponent<Text>();
        t.transform.SetParent(go.transform, false);
        t.text = label;
        t.font = Resources.GetBuiltinResource<Font>("LegacyRuntime.ttf");
        t.alignment = TextAnchor.MiddleCenter;
        t.color = Color.white; t.fontSize = 11;
        
        var outline = go.AddComponent<Outline>();
        outline.effectColor = kColorBorder;
        outline.effectDistance = new Vector2(1, -1);
    }

    void FindAndLinkUI()
    {
        var root = GameObject.Find("SidebarContent");
        if (root == null) return;

        focalSlider = root.transform.Find("SLIDER_FOCAL_ROOT/SLIDER_FOCAL")?.GetComponent<Slider>();
        focalInput = root.transform.Find("SLIDER_FOCAL_ROOT/Header/ValueRow/Input")?.GetComponent<InputField>();
        
        panSlider = root.transform.Find("SLIDER_PAN_ROOT/SLIDER_PAN")?.GetComponent<Slider>();
        panInput = root.transform.Find("SLIDER_PAN_ROOT/Header/ValueRow/Input")?.GetComponent<InputField>();
        
        tiltSlider = root.transform.Find("SLIDER_TILT_ROOT/SLIDER_TILT")?.GetComponent<Slider>();
        tiltInput = root.transform.Find("SLIDER_TILT_ROOT/Header/ValueRow/Input")?.GetComponent<InputField>();
        
        rollSlider = root.transform.Find("SLIDER_ROLL_ROOT/SLIDER_ROLL")?.GetComponent<Slider>();
        rollInput = root.transform.Find("SLIDER_ROLL_ROOT/Header/ValueRow/Input")?.GetComponent<InputField>();
        
        distSlider = root.transform.Find("SLIDER_DIST_ROOT/SLIDER_DIST")?.GetComponent<Slider>();
        distInput = root.transform.Find("SLIDER_DIST_ROOT/Header/ValueRow/Input")?.GetComponent<InputField>();

        var actions = root.transform.Find("ActionsRow");
        if (actions) {
            resetButton = actions.Find("BTN_RESET")?.GetComponent<Button>();
            captureButton = actions.Find("BTN_CAPTURE")?.GetComponent<Button>();
        }

        string[] labels = {"ORG", "16:9", "4:3", "1:1", "9:16"};
        aspectButtons = new Button[labels.Length];
        var aspectRow = root.transform.Find("AspectRow");
        if (aspectRow) {
            for(int i=0; i<labels.Length; i++) {
                aspectButtons[i] = aspectRow.Find("BTN_ASPECT_" + labels[i])?.GetComponent<Button>();
            }
        }

        if (resetButton) { resetButton.onClick.RemoveAllListeners(); resetButton.onClick.AddListener(() => { if (splatRenderer && splatRenderer.m_Asset) ResetCamera(splatRenderer.m_Asset); }); }
        if (captureButton) { captureButton.onClick.RemoveAllListeners(); captureButton.onClick.AddListener(CaptureScreenshot); }

        SetupCameraUI();
        LogToConsole("Sidebar Stylized & Verified.");
    }

    void SetupCameraUI()
    {
        if (focalSlider) { focalSlider.onValueChanged.RemoveAllListeners(); focalSlider.onValueChanged.AddListener(val => ApplyFocalFromUI(val)); }
        if (panSlider) { panSlider.onValueChanged.RemoveAllListeners(); panSlider.onValueChanged.AddListener(val => ApplyPanFromUI(val)); }
        if (tiltSlider) { tiltSlider.onValueChanged.RemoveAllListeners(); tiltSlider.onValueChanged.AddListener(val => ApplyTiltFromUI(val)); }
        if (rollSlider) { rollSlider.onValueChanged.RemoveAllListeners(); rollSlider.onValueChanged.AddListener(val => ApplyRollFromUI(val)); }
        if (distSlider) { distSlider.onValueChanged.RemoveAllListeners(); distSlider.onValueChanged.AddListener(val => ApplyDistFromUI(val)); }

        if (focalInput) { focalInput.onEndEdit.RemoveAllListeners(); focalInput.onEndEdit.AddListener(val => { if(float.TryParse(val, out float f)) { focalSlider.value = f; ApplyFocalFromUI(f); } }); }
        if (panInput) { panInput.onEndEdit.RemoveAllListeners(); panInput.onEndEdit.AddListener(val => { if(float.TryParse(val, out float f)) { panSlider.value = f; ApplyPanFromUI(f); } }); }
        if (tiltInput) { tiltInput.onEndEdit.RemoveAllListeners(); tiltInput.onEndEdit.AddListener(val => { if(float.TryParse(val, out float f)) { tiltSlider.value = f; ApplyTiltFromUI(f); } }); }
        if (rollInput) { rollInput.onEndEdit.RemoveAllListeners(); rollInput.onEndEdit.AddListener(val => { if(float.TryParse(val, out float f)) { rollSlider.value = f; ApplyRollFromUI(f); } }); }
        if (distInput) { distInput.onEndEdit.RemoveAllListeners(); distInput.onEndEdit.AddListener(val => { if(float.TryParse(val, out float f)) { distSlider.value = f; ApplyDistFromUI(f); } }); }

        if (aspectButtons != null)
        {
            foreach (var btn in aspectButtons)
            {
                if (btn == null) continue;
                string label = btn.GetComponentInChildren<Text>()?.text;
                btn.onClick.RemoveAllListeners();
                btn.onClick.AddListener(() => SetAspectRatio(label));
            }
        }
    }

    void ApplyFocalFromUI(float val) { if(isUpdatingUI) return; float vFov = 2.0f * Mathf.Atan(24.0f / (2.0f * val)) * Mathf.Rad2Deg; if(mainCamera) mainCamera.fieldOfView = vFov; SyncLabels(); }
    void ApplyPanFromUI(float val) { if(isUpdatingUI) return; if(cameraController) { cameraController.Pan = val; cameraController.UpdateCameraTransform(); } SyncLabels(); }
    void ApplyTiltFromUI(float val) { if(isUpdatingUI) return; if(cameraController) { cameraController.Tilt = val; cameraController.UpdateCameraTransform(); } SyncLabels(); }
    void ApplyRollFromUI(float val) { if(isUpdatingUI) return; if(cameraController) { cameraController.Roll = val; cameraController.UpdateCameraTransform(); } SyncLabels(); }
    void ApplyDistFromUI(float val) { if(isUpdatingUI) return; if(cameraController) { cameraController.distance = val; cameraController.UpdateCameraTransform(); } SyncLabels(); }

    void SyncLabels()
    {
        isUpdatingUI = true;
        if (focalInput && !focalInput.isFocused && focalSlider) focalInput.text = focalSlider.value.ToString("F1");
        if (panInput && !panInput.isFocused && panSlider) panInput.text = panSlider.value.ToString("F1");
        if (tiltInput && !tiltInput.isFocused && tiltSlider) tiltInput.text = tiltSlider.value.ToString("F1");
        if (rollInput && !rollInput.isFocused && rollSlider) rollInput.text = rollSlider.value.ToString("F1");
        if (distInput && !distInput.isFocused && distSlider) distInput.text = distSlider.value.ToString("F1");
        isUpdatingUI = false;
    }

    void SetAspectRatio(string ratio)
    {
        if (mainCamera == null) return;
        float targetAspect = 1.0f;
        switch (ratio)
        {
            case "16:9": targetAspect = 16f / 9f; break;
            case "4:3": targetAspect = 4f / 3f; break;
            case "1:1": targetAspect = 1f; break;
            case "9:16": targetAspect = 9f / 16f; break;
            case "ORG": targetAspect = currentMetadata != null ? currentMetadata.aspect_ratio : 1.0f; break;
            default: return;
        }

        float currentAspect = (float)Screen.width / Screen.height;
        if (targetAspect > currentAspect)
        {
            float inset = currentAspect / targetAspect;
            mainCamera.rect = new Rect(0, (1 - inset) / 2, 1, inset);
        }
        else
        {
            float inset = targetAspect / currentAspect;
            mainCamera.rect = new Rect((1 - inset) / 2, 0, inset, 1);
        }
        LogToConsole($"Aspect Ratio set to {ratio} ({targetAspect:F2})");
    }

    void OnDestroy()
    {
        if (backendProcess != null && !backendProcess.HasExited)
        {
            backendProcess.Kill();
            backendProcess.Dispose();
        }
    }

    void LogToConsole(string msg)
    {
        UnityEngine.Debug.Log("[Viewer] " + msg);
        lock (logLock)
        {
            consoleLines.Add($"[{System.DateTime.Now:HH:mm:ss}] {msg}");
            if (consoleLines.Count > 50) consoleLines.RemoveAt(0);
        }
    }

    void UpdateStatusText()
    {
        if (statusText != null) 
            statusText.text = "L-Click: Rotate | R-Click: Pan | Scroll: Zoom\nKeys: [R] Rotate 90 | [X] Mirror X | [Y] Mirror Y";
    }

    void Update()
    {
        if (triggerUICreation)
        {
            triggerUICreation = false;
            CreateSidebarUI();
        }

        lock (logLock)
        {
            if (debugConsoleText != null)
                debugConsoleText.text = string.Join("\n", consoleLines);
        }

        if (Application.isPlaying)
        {
            if (isGenerating)
            {
                if (statusText != null) statusText.text = "AI GENERATING MODEL...";
            }
            else if (!isUpdatingUI && !Mouse.current.leftButton.isPressed && !Mouse.current.rightButton.isPressed) 
            {
                SyncUIFromCamera();
            }

            var keyboard = Keyboard.current;
            if (keyboard == null) return;

            if (keyboard.rKey.wasPressedThisFrame) RotateModel();
            if (keyboard.xKey.wasPressedThisFrame) MirrorModel(true, false);
            if (keyboard.yKey.wasPressedThisFrame) MirrorModel(false, true);
        }
    }

    void SyncUIFromCamera()
    {
        if (cameraController == null || isUpdatingUI) return;
        
        isUpdatingUI = true;

        if (panSlider) panSlider.value = cameraController.Pan;
        if (panInput && !panInput.isFocused) panInput.text = cameraController.Pan.ToString("F1");
        
        if (tiltSlider) tiltSlider.value = cameraController.Tilt;
        if (tiltInput && !tiltInput.isFocused) tiltInput.text = cameraController.Tilt.ToString("F1");

        if (rollSlider) rollSlider.value = cameraController.Roll;
        if (rollInput && !rollInput.isFocused) rollInput.text = cameraController.Roll.ToString("F1");

        if (distSlider) distSlider.value = cameraController.distance;
        if (distInput && !distInput.isFocused) distInput.text = cameraController.distance.ToString("F1");

        if (mainCamera != null && focalSlider)
        {
            float vFov = mainCamera.fieldOfView;
            float focal = 24.0f / (2.0f * Mathf.Tan(vFov * Mathf.Deg2Rad / 2.0f));
            focalSlider.value = focal;
            if (focalInput && !focalInput.isFocused) focalInput.text = focal.ToString("F1");
        }

        isUpdatingUI = false;
    }

    public void RotateModel()
    {
        if (splatRenderer != null)
        {
            splatRenderer.transform.Rotate(90, 0, 0);
            LogToConsole("Rotated model 90 deg on X.");
        }
    }

    public void MirrorModel(bool flipX, bool flipY)
    {
        if (splatRenderer != null)
        {
            Vector3 scale = splatRenderer.transform.localScale;
            if (flipX) scale.x *= -1;
            if (flipY) scale.y *= -1;
            splatRenderer.transform.localScale = scale;
            LogToConsole("Mirrored model. New scale: " + scale);
        }
    }

    void OnLoadButtonClicked()
    {
#if UNITY_EDITOR
        string path = EditorUtility.OpenFilePanel("Select Gaussian Splat (PLY or Asset)", "Assets", "ply,asset");
        if (!string.IsNullOrEmpty(path))
        {
            if (path.ToLower().EndsWith(".ply")) AutoCreateAndLoadAsset(path);
            else if (path.ToLower().EndsWith(".asset")) LoadAssetByPath(path);
        }
#endif
    }

    void OnGenerateButtonClicked()
    {
#if UNITY_EDITOR
        string path = EditorUtility.OpenFilePanel("Select Image for Generation", "", "jpg,png,jpeg");
        if (!string.IsNullOrEmpty(path))
        {
            StartCoroutine(GenerateAndLoad(path));
        }
#endif
    }

    IEnumerator GenerateAndLoad(string imagePath)
    {
        isGenerating = true;
        if (generateButton != null) generateButton.interactable = false;
        LogToConsole("Starting generation process...");

        UnityWebRequest checkReq = UnityWebRequest.Get(BackendUrl + "/history");
        yield return checkReq.SendWebRequest();

        if (checkReq.result != UnityWebRequest.Result.Success)
        {
            LogToConsole("Backend not running. Attempting to start...");
            StartBackend();
            
            float timeout = 25f;
            while (timeout > 0)
            {
                UnityWebRequest retry = UnityWebRequest.Get(BackendUrl + "/history");
                yield return retry.SendWebRequest();
                if (retry.result == UnityWebRequest.Result.Success) break;
                timeout -= 2f;
                yield return new WaitForSeconds(2f);
            }
            if (timeout <= 0) 
            { 
                LogToConsole("Server failed to start."); 
                isGenerating = false;
                if (generateButton != null) generateButton.interactable = true;
                yield break; 
            }
            LogToConsole("Server connected.");
        }
        else
        {
            LogToConsole("Connected to existing backend.");
        }

        LogToConsole("Uploading image...");

        byte[] imageBytes = File.ReadAllBytes(imagePath);
        WWWForm form = new WWWForm();
        form.AddBinaryData("file", imageBytes, Path.GetFileName(imagePath), "image/jpeg");

        using (UnityWebRequest www = UnityWebRequest.Post(BackendUrl + "/generate", form))
        {
            www.timeout = 1200; 
            var op = www.SendWebRequest();
            
            while (!op.isDone)
            {
                if (www.uploadProgress < 1f)
                    statusText.text = $"Uploading... {www.uploadProgress * 100:F0}%";
                else
                    statusText.text = "Server generating model... check console below.";
                yield return null;
            }

            isGenerating = false;
            if (generateButton != null) generateButton.interactable = true;

            if (www.result != UnityWebRequest.Result.Success)
            {
                LogToConsole("ERROR: Generation failed: " + www.error);
                yield break;
            }

            string json = www.downloadHandler.text;
            LogToConsole("Generation Complete. Parsing response...");
            
            string fileId = "";
            try
            {
                GenerateResponse res = JsonUtility.FromJson<GenerateResponse>(json);
                fileId = res.id;
                currentMetadata = res.metadata;
                LogToConsole("Parsed ID: " + fileId + " | Focal: " + (currentMetadata != null ? currentMetadata.focal_length_35mm.ToString() : "N/A"));
            }
            catch (System.Exception e)
            {
                LogToConsole("JSON parse error: " + e.Message + " | Raw: " + json);
            }

            if (!string.IsNullOrEmpty(fileId))
            {
                yield return StartCoroutine(DownloadAndLoadPly(fileId));
            }
            else
            {
                LogToConsole("ERROR: File ID is empty. Server response: " + json);
            }
        }
    }


    IEnumerator DownloadAndLoadPly(string fileId)
    {
        LogToConsole("Downloading generated PLY from server...");
        string downloadUrl = $"{BackendUrl}/output/{fileId}/model.ply";
        
        using (UnityWebRequest www = UnityWebRequest.Get(downloadUrl))
        {
            yield return www.SendWebRequest();

            if (www.result != UnityWebRequest.Result.Success)
            {
                LogToConsole("Download Failed: " + www.error);
                yield break;
            }

            string downloadPath = Path.Combine(Application.dataPath, "GaussianAssets", "Downloads");
            if (!Directory.Exists(downloadPath)) Directory.CreateDirectory(downloadPath);
            
            string localPlyPath = Path.Combine(downloadPath, $"{fileId}.ply");
            File.WriteAllBytes(localPlyPath, www.downloadHandler.data);
            
            LogToConsole("File saved locally. Starting optimization...");
            
            yield return new WaitForSeconds(1f);
            
#if UNITY_EDITOR
            AutoCreateAndLoadAsset(localPlyPath);
#endif
        }
    }

    void StartBackend()
    {
        try
        {
            backendProcess = new Process();
            
            if (File.Exists(PythonExePath) && File.Exists(ServerScriptPath))
            {
                LogToConsole("Starting backend via Python (Dev Mode)...");
                backendProcess.StartInfo.FileName = PythonExePath;
                backendProcess.StartInfo.Arguments = ServerScriptPath;
                backendProcess.StartInfo.WorkingDirectory = Path.GetDirectoryName(ServerScriptPath);
            }
            else
            {
                LogToConsole("Starting backend via EXE (Production Mode)...");
                backendProcess.StartInfo.FileName = BackendExePath;
                backendProcess.StartInfo.WorkingDirectory = Path.GetDirectoryName(BackendExePath);
            }
            
            backendProcess.StartInfo.UseShellExecute = false;
            backendProcess.StartInfo.RedirectStandardOutput = true;
            backendProcess.StartInfo.RedirectStandardError = true;
            backendProcess.StartInfo.CreateNoWindow = true;

            backendProcess.OutputDataReceived += (sender, e) => { if (e.Data != null) LogToConsole("[EXE] " + e.Data); };
            backendProcess.ErrorDataReceived += (sender, e) => { if (e.Data != null) LogToConsole("[EXE-ERR] " + e.Data); };

            backendProcess.Start();
            backendProcess.BeginOutputReadLine();
            backendProcess.BeginErrorReadLine();
        }
        catch (System.Exception e)
        {
            LogToConsole("Process Start Failed: " + e.Message);
        }
    }

    void AutoCreateAndLoadAsset(string plyPath)
    {
#if UNITY_EDITOR
        string fileName = Path.GetFileNameWithoutExtension(plyPath);
        string pathHash = plyPath.GetHashCode().ToString("X");
        string uniqueName = $"{fileName}_{pathHash}";
        
        string targetFolder = "Assets/GaussianAssets";
        if (!Directory.Exists(targetFolder)) Directory.CreateDirectory(targetFolder);
        string targetAssetPath = $"{targetFolder}/{uniqueName}.asset";

        var creator = ScriptableObject.CreateInstance<GaussianSplatAssetCreator>();
        var type = creator.GetType();
        type.GetField("m_InputFile", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance).SetValue(creator, plyPath);
        type.GetField("m_OutputFolder", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance).SetValue(creator, targetFolder);
        
        var applyQualityMethod = type.GetMethod("ApplyQualityLevel", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
        applyQualityMethod?.Invoke(creator, null);

        var createMethod = type.GetMethod("CreateAsset", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
        if (createMethod != null)
        {
            createMethod.Invoke(creator, null);
            AssetDatabase.Refresh();
            
            string defaultCreatedPath = $"{targetFolder}/{fileName}.asset";
            if (File.Exists(defaultCreatedPath))
            {
                if (File.Exists(targetAssetPath)) AssetDatabase.DeleteAsset(targetAssetPath);
                AssetDatabase.MoveAsset(defaultCreatedPath, targetAssetPath);
            }

            DestroyImmediate(creator);
            LoadAssetByPath(targetAssetPath);
        }
#endif
    }

    void LoadAssetByPath(string path)
    {
#if UNITY_EDITOR
        if (path.StartsWith(Application.dataPath)) path = "Assets" + path.Substring(Application.dataPath.Length);
        GaussianSplatAsset asset = AssetDatabase.LoadAssetAtPath<GaussianSplatAsset>(path);
        if (asset != null) LoadAsset(asset);
        else LogToConsole("Load Failed: " + path);
#endif
    }

    public void LoadAsset(GaussianSplatAsset asset)
    {
        if (splatRenderer != null)
        {
            splatRenderer.m_Asset = asset;
            splatRenderer.transform.localScale = new Vector3(1, -1, 1);
            splatRenderer.transform.rotation = Quaternion.identity;
            splatRenderer.enabled = false;
            splatRenderer.enabled = true;

            UpdateStatusText();
            LogToConsole("DONE: Model Loaded.");
            
            ResetCamera(asset);
        }
    }

    public void ResetCamera(GaussianSplatAsset asset)
    {
        if (cameraController != null && cameraController.target != null)
        {
            isUpdatingUI = true;

            // Apex of the cone (Origin)
            Vector3 worldOrigin = splatRenderer.transform.TransformPoint(Vector3.zero);
            
            cameraController.target.position = worldOrigin;
            cameraController.distance = 5.0f;

            cameraController.SetRotation(0, 0, 0);
            cameraController.UpdateCameraTransform();

            if (mainCamera != null && currentMetadata != null)
            {
                float focalLength = currentMetadata.focal_length_35mm;
                if (focalLength <= 0) focalLength = 35.0f; 
                float vFov = 2.0f * Mathf.Atan(24.0f / (2.0f * focalLength)) * Mathf.Rad2Deg;
                mainCamera.fieldOfView = vFov;
            }

            isUpdatingUI = false;
            SyncUIFromCamera();
            LogToConsole("Camera reset to Apex View.");
        }
    }

    void CaptureScreenshot()
    {
        string folder = Path.Combine(Application.dataPath, "Captures");
        if (!Directory.Exists(folder)) Directory.CreateDirectory(folder);
        string filename = $"Splat_{System.DateTime.Now:yyyyMMdd_HHmmss}.png";
        ScreenCapture.CaptureScreenshot(Path.Combine(folder, filename));
        LogToConsole("Screenshot saved to Assets/Captures/" + filename);
    }
}
