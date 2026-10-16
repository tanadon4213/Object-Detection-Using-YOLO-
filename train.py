from ultralytics import YOLO

def main():
    print("=== เริ่มต้นทำแลป Object Detection Using YOLO ===")

    # 1. โหลดโมเดลสำเร็จรูป (YOLOv8 Nano)
    model = YOLO('yolov8n.pt')

    # -------------------------------------------------------------
    # 2. การเทรนรอบที่ 1 (Configuration 1: ตัวอย่างเช่น 20 Epochs)
    # *หมายเหตุ: เปลี่ยน data='path/to/data.yaml' เป็นไฟล์ data.yaml ของคุณ
    # -------------------------------------------------------------
    print("\n[1/3] กำลังเทรนโมเดล Config ที่ 1 (20 Epochs)...")
    results_1 = model.train(
        data='coco8.yaml',  # <-- เปลี่ยนตรงนี้เป็น path ไฟล์ .yaml ของชุดข้อมูลคุณ
        epochs=20,          
        imgsz=640,          
        batch=16,           
        name='lab_run_epoch20'
    )

    # -------------------------------------------------------------
    # 3. การเทรนรอบที่ 2 (Configuration 2: เปรียบเทียบด้วย 50 Epochs ตามโจทย์)
    # -------------------------------------------------------------
    print("\n[2/3] กำลังเทรนโมเดล Config ที่ 2 (50 Epochs เพื่อเปรียบเทียบ)...")
    results_2 = model.train(
        data='coco8.yaml',  # <-- เปลี่ยนตรงนี้เป็น path ไฟล์ .yaml ของชุดข้อมูลคุณ
        epochs=50,          
        imgsz=640,          
        batch=16,           
        name='lab_run_epoch50'
    )

    # 4. ประเมินผลโมเดล (Evaluation) เพื่อดึงค่า mAP, Precision, Recall
    print("\n[3/3] กำลังประเมินผลประสิทธิภาพโมเดล (Evaluation)...")
    metrics = model.val()
    
    print("\n--- ผลลัพธ์การประเมิน (Metrics) ---")
    print(f"Precision: {metrics.box.p}")
    print(f"Recall: {metrics.box.r}")
    print(f"mAP@0.5: {metrics.box.map50}")
    print(f"mAP@0.5:0.95: {metrics.box.map}")

    # 5. ทดสอบโมเดลกับภาพตัวอย่าง (Inference)
    # สามารถเปลี่ยนลิงก์ภาพเป็น path รูปภาพจริงในเครื่องของคุณได้ เช่น 'my_test_image.jpg'
    test_image_path = 'https://ultralytics.com/images/bus.jpg'
    detect_results = model(test_image_path, save=True, conf=0.5)

    print("\n🎉 ทำแลปเสร็จสมบูรณ์! ผลลัพธ์ทั้งหมดถูกบันทึกไว้ในโฟลเดอร์ 'runs/detect/' แล้ว")

if __name__ == '__main__':
    main()