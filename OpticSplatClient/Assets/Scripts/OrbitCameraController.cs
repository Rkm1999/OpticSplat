using UnityEngine;
using UnityEngine.InputSystem;
using UnityEngine.EventSystems;

public class OrbitCameraController : MonoBehaviour
{
    public Transform target;
    public float distance = 5.0f;
    public float xSpeed = 120.0f;
    public float ySpeed = 120.0f;
    public float scrollSpeed = 10.0f;
    public float panSpeed = 0.5f;

    public float yMinLimit = -80f;
    public float yMaxLimit = 80f;

    public float distanceMin = .1f;
    public float distanceMax = 1000f;

    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f; // Roll

    public float Pan { get { return x; } set { x = value; } }
    public float Tilt { get { return y; } set { y = value; } }
    public float Roll { get { return z; } set { z = value; } }

    void Start()
    {
        Vector3 angles = transform.eulerAngles;
        x = angles.y;
        y = angles.x;
        z = angles.z;

        if (GetComponent<Rigidbody>())
        {
            GetComponent<Rigidbody>().freezeRotation = true;
        }
    }

    void LateUpdate()
    {
        if (target)
        {
            Mouse mouse = Mouse.current;
            if (mouse != null)
            {
                // ONLY process mouse input if NOT hovering over UI
                if (!IsPointerOverUI())
                {
                    // Rotation (Orbit around target)
                    if (mouse.leftButton.isPressed)
                    {
                        Vector2 delta = mouse.delta.ReadValue();
                        x += delta.x * xSpeed * 0.02f;
                        y -= delta.y * ySpeed * 0.02f;
                        y = ClampAngle(y, yMinLimit, yMaxLimit);
                    }

                    // Panning (Moves target relative to view)
                    if (mouse.rightButton.isPressed)
                    {
                        Vector2 delta = mouse.delta.ReadValue();
                        Vector3 move = transform.right * (-delta.x * panSpeed * 0.01f * (distance * 0.1f)) + 
                                       transform.up * (-delta.y * panSpeed * 0.01f * (distance * 0.1f));
                        target.position += move;
                    }

                    // Zooming (Changes orbit distance)
                    float scroll = mouse.scroll.ReadValue().y;
                    if (Mathf.Abs(scroll) > 0.01f)
                    {
                        distance = Mathf.Clamp(distance - scroll * scrollSpeed * 0.01f * (distance * 0.1f), distanceMin, distanceMax);
                    }
                }
            }

            UpdateCameraTransform();
        }
    }

    private bool IsPointerOverUI()
    {
        if (EventSystem.current == null) return false;
        return EventSystem.current.IsPointerOverGameObject();
    }

    public void UpdateCameraTransform()
    {
        if (!target) return;

        Quaternion rotation = Quaternion.Euler(y, x, z);
        Vector3 position = target.position - (rotation * Vector3.forward * distance);

        transform.rotation = rotation;
        transform.position = position;
    }

    public static float ClampAngle(float angle, float min, float max)
    {
        if (angle < -360F)
            angle += 360F;
        if (angle > 360F)
            angle -= 360F;
        return Mathf.Clamp(angle, min, max);
    }

    public void SetRotation(float pan, float tilt, float roll = 0)
    {
        x = pan;
        y = tilt;
        z = roll;
    }
}
