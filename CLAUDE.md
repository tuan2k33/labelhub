# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Mục đích

LabelHub: platform gán nhãn box → version ảnh/nhãn/model → train YOLO → inference, một người dùng, chạy trên GPU 4 GB. SAM3 tự gán nhãn, người chỉ sửa (Manual) hoặc bỏ qua (Auto). Ghi chú thiết kế (đã làm gì, vì sao, và cái gì bị loại) nằm ở `SPEC.md`; README.md là bản giới thiệu cho người đọc repo; cả hai viết bằng tiếng Anh. Giấy phép: Apache-2.0. Còn lại chưa làm: backend `sam3-trt`, vòng self-training bằng YOLO (người dùng đã quyết định SAM3 gán nhãn là đủ, không ưu tiên TensorRT).

## Lệnh

Venv ở `.venv` (symlink tới `~/.dev/venvs/labelhub`, Python 3.14, PyTorch nightly cu134). Cài gói: `uv pip install --python .venv/bin/python <gói>`; danh sách tên gói ở `requirements.txt`.

- Test: `.venv/bin/python -m pytest -q`; một test: `.venv/bin/python -m pytest -q tests/test_auto.py::test_gold_needs_review_and_is_never_trained_on`. `tests/test_inference.py` cần `~/.dev/data/lh_coco/weights/yolo26n.pt` (không có thì tự skip) và mất ~40 giây vì nạp ultralytics. Không có lint.
- Chạy app: `LABELHUB_DATA=<thư mục> .venv/bin/uvicorn labelhub.projects:create_server --factory --port 8000`, mở http://localhost:8000 (API ở `/api`, giao diện đã build ở `/`). SAM3 cần `HF_TOKEN` trong `.env` và quyền truy cập `facebook/sam3` (model bị khoá).
- Giao diện: `cd frontend && npm run build` (tsc + vite, server tự phục vụ `frontend/dist`); `npm run dev` (proxy `/api` sang 8000).
- Bản demo tĩnh cho Claude artifact: `npm run build:demo && python frontend/scripts/make_artifact.py`. Dữ liệu mẫu `frontend/src/demo/data.json` (chứa ảnh COCO nhúng base64, đã gitignore, không được commit) do `scripts/make_demo_data.py` và `scripts/add_demo_predictions.py` sinh từ một server thật; `npm run build` thường loại `src/demo` nên không cần file này.
- Script thí nghiệm: `PYTHONPATH=. .venv/bin/python experiments/<tên>.py` (đọc dữ liệu COCO ở `~/.dev/data`).

## Kiến trúc

**Dataset = một thư mục dữ liệu độc lập.** `projects.Registry` quét `<root>/projects/<slug>/` (mỗi nơi có `labelhub.db`, `images/`, `models/`, `thumbs/`, `meta.json`); một `labelhub.db` nằm thẳng trong root (bố cục cũ) hiện thành dataset `default`. `DatasetRouter` mount app FastAPI của từng dataset dưới `/api/p/<slug>` và đặt thư mục dữ liệu của request qua ContextVar (`config.use_data_dir`); `config.data_dir()` và `images_dir()` đọc nó. Thread tự tạo trong request phải chạy bằng `contextvars.copy_context().run`, còn tiến trình train nhận `LABELHUB_DATA` qua biến môi trường. Mọi dataset dùng chung một dict `shared` (SAM3, khoá GPU, khoá Auto, bảng job) và `probe` để biết có lần train nào đang chạy ở dataset khác.

**Dữ liệu và version** (`db.py`, `versions.py`). Ảnh định danh bằng sha256 và bất biến. `ImageVersion` = tập ảnh + split (train/val/test/gold, chia theo hash+seed nên ổn định khi thêm ảnh); `LabelVersion` = box `accepted` + class set + thống kê nguồn nhãn; `ModelVersion` lưu cặp (image, label) đã train. Tên version là `YYYY.MM.x`, đếm riêng cho từng bảng. Dữ liệu train = ImageVersion ∩ LabelVersion; ảnh gold không bao giờ vào train mà được chấm riêng. LabelVersion mặc định chỉ lấy ảnh đã `reviewed`; Auto dùng `include_unopened` để lấy cả ảnh SAM3 đã gán (`auto_done`).

**Quy ước review** (người dùng đã chốt): mở ảnh = đã duyệt (`POST /images/{hash}/view`), không có nút tích duyệt hay ✓/✕ từng box. Thanh trượt 0.3–1.0 ở Editor ghi lại `status` của box auto (`accepted`/`rejected`, `rejected` là ẩn nhưng còn lưu) nên nhãn luôn đúng với cái đang thấy. Auto-label nhận mặc định ở 0.5; sàn 0.3 là tham số `Sam3Detector(threshold=0.3)` của code mình (cũng là mặc định của transformers).

**Kỷ luật GPU 4 GB.** Chỉ một job GPU một lúc. `autolabel.Sam3Detector` chạy fp16, mã hoá ảnh một lần rồi dùng lại cho mọi class, và phải được giải phóng (`pipeline.close_detector`) trước khi train, nếu không train dễ hết VRAM. Train là tiến trình riêng `labelhub.train_worker` (nhóm tiến trình riêng, huỷ = SIGTERM cả nhóm); hàng `ModelVersion` trong DB là nguồn sự thật, worker chết thì `training.refresh` đánh dấu failed. Suy luận (`inference.py`) tự chạy trên CPU khi đang train. Chuỗi Auto (`pipeline.py`) chạy trong thread của server và trạng thái chỉ nằm trong bộ nhớ.

**Giao diện** (`frontend/`, React 19 + Vite + TS, không router: hash `#<slug>` chọn dataset). `api.ts` giữ base path hiện tại qua `setDataset()`. `flags.demo` + `demo/mock.ts` thay toàn bộ API bằng bản giả chạy trong trình duyệt cho artifact.

## Lưu ý

- `frontend/node_modules` là symlink sang `~/.dev/nm-work/labelhub-frontend/node_modules`. Đừng `npm install` trong `frontend/`: npm thay symlink bằng thư mục thật trên `/mnt/c` và thiếu gói native. Muốn thêm gói: cài trong thư mục ext4 kia rồi chép `package.json` và `package-lock.json` về.
- Đổi cột DB: `create_all` không ALTER, thêm cột mới vào `db._migrate`.
- Python 3.14 dùng forkserver cho multiprocessing: script nào gọi `ultralytics train` phải có `if __name__ == "__main__"`.
- Đối tượng SQLModel trả về sau `commit` bị expire, serialize ra `{}` nếu không `refresh`/`model_dump` ngay.
- Test chạm tới `training.cancel` phải dùng tiến trình giả riêng (ví dụ `sleep`); đừng trả `os.getpid()`, lệnh huỷ sẽ giết pytest.
- Artifact không cho tải file qua `<a download>`; bản demo vô hiệu các nút tải và train.
- `data/`, `output/`, `weights/`, `runs/`, `*.pt`, `.env`, `frontend/src/demo/data.json` đã ignore. Không commit weights hay ảnh dataset (ảnh COCO có giấy phép riêng); ảnh chụp trong `docs/screenshots` chỉ được dùng ảnh public domain/CC0 (nguồn ghi ở `docs/screenshots/CREDITS.md`), thêm ảnh mới thì thêm dòng ghi nguồn.
