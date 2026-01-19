import cv2
from deepface import DeepFace

emoji_map = {
    'mutlu': '😊',
    'üzgün': '😢',
    'kızgın': '😠',
    'natural': '😐'
}

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Kamera görüntüsü alınamadı.")
        break

    frame = cv2.flip(frame, 1)

    try:

        results = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)

    
        if isinstance(results, list):
            for result in results:
                x, y, w, h = result['region']['x'], result['region']['y'], result['region']['w'], result['region']['h']
                emotion = result['dominant_emotion']
                emoji = emoji_map.get(emotion, '')
                label = f"{emotion} {emoji}"
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
        else:
   
            x, y, w, h = results['region']['x'], results['region']['y'], results['region']['w'], results['region']['h']
            emotion = results['dominant_emotion']
            emoji = emoji_map.get(emotion, '')
            label = f"{emotion} {emoji}"
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

    except Exception as e:
        print(f"Hata: {e}")

    cv2.imshow("Gerçek Zamanlı Duygu Analizi", frame)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
