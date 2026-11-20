# 설치와 실행

계산은 로컬에서, 화면은 브라우저로 봅니다. 같은 코드가 WSL·Linux(NVIDIA CUDA), Mac(Apple Metal), CPU 에서 돕니다.

## 공통

```bash
mkdir -p ~/sim && cd ~/sim
git clone https://github.com/MunkyunLEE/Simweb-Core.git
git clone https://github.com/MunkyunLEE/SPH-LLM-web.git
cd SPH-LLM-web
./scripts/setup.sh
./scripts/run.sh
```

- Python 3.10 ~ 3.13 (Taichi 지원 범위). 가상환경은 `~/.venvs/sphweb` (`SPHWEB_VENV` 로 변경)
- 두 저장소를 같은 폴더 아래 나란히 두면 Simweb-Core 를 고친 내용이 바로 반영됩니다 (editable 설치)
- 설치 마지막에 계산 장치 확인 결과와 간이 벤치마크가 나옵니다
- 포트 변경: `SPHWEB_PORT=8800 ./scripts/run.sh`

## WSL2 Ubuntu (회사 PC)

- 작업 폴더는 WSL 홈 아래에 둡니다. `/mnt/c/...` 는 파일 입출력이 몇 배 느립니다
- NVIDIA GPU: Windows 쪽에 드라이버만 있으면 되고 WSL 안에 드라이버를 따로 설치하지 않습니다. Ubuntu 안에서 `nvidia-smi` 가 GPU 를 보여 주면 Taichi 가 CUDA 를 씁니다
- Vulkan 은 WSL 에서 불안정하므로 쓰지 않습니다 (자동 선택은 cuda → cpu)
- 사내 프록시·인증서: FEM-LLM-web 의 `docs/WSL.md` 와 같은 설정을 쓰고, 인증서 묶음은 `SPHWEB_CA_BUNDLE` 로 넘깁니다
- Windows 브라우저가 자동으로 열리지 않으면 `http://localhost:8766` 을 직접 엽니다

## Mac (Apple Silicon)

- `brew install python@3.12` 등으로 3.10 ~ 3.13 Python 을 준비하고 `SPHWEB_PYTHON=python3.12 ./scripts/setup.sh`
- Taichi 가 Metal 을 잡으면 GPU 로 계산합니다 (자동 선택은 metal → cpu)

## 계산 장치 확인만 다시 하기

```bash
. ~/.venvs/sphweb/bin/activate
python -m sphweb.engine.backend --bench 2000000
```

## LLM

설치할 것은 없습니다. 화면 오른쪽 위 LLM 칸에 OpenAI 호환 엔드포인트(`.../v1`), 모델 이름, API 키를 넣고 "적용 · 확인", "실제 응답 테스트" 를 누릅니다. 시작할 때부터 연결하려면 `SPHWEB_LLM_URL`, `SPHWEB_LLM_MODEL`, `SPHWEB_LLM_KEY`, `SPHWEB_LLM_MODE`(rule_first 권장).
