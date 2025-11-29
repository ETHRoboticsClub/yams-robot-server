## Quick Start

```bash
git clone --recursive https://github.com/ETHRoboticsClub/yams-robot-server.git
cd yams-robot-server
```

## Installation

### Virtual Environment (Recommended)

First, install uv if you don't have it:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```
Now install the dependencies:
```bash
uv sync
```

## Usage

- **Follower arms:** Check CAN port mapping by plugging cables one at a time and monitoring `ip link show`
- **Leader arms:** Check USB port mapping by plugging cables one at a time and monitoring `ls /dev/ttyACM*`

### Reset CAN Devices
```bash
sh third_party/i2rt/scripts/reset_all_can.sh
```
### Run bimanual teleoperation script
**Important:** Put the leader arms into a nominal position before starting the script.
```bash
uv run examples/bi_leader_follower.py
```