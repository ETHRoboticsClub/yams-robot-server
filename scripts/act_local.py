#!/usr/bin/env python3
import argparse
import shlex
import subprocess

DEFAULT_CAMERAS = "{left_wrist: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}, right_wrist: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30}}"


def run(cmd: list[str], dry_run: bool) -> None:
    print("$", " ".join(shlex.quote(x) for x in cmd))
    if not dry_run:
        subprocess.run(cmd, check=True)


def base_robot_args(args: argparse.Namespace) -> list[str]:
    return [
        f"--robot.type={args.robot_type}",
        f"--robot.cameras={args.cameras}",
    ]


def add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--robot-type", default="bi_yams_follower")
    parser.add_argument("--cameras", default=DEFAULT_CAMERAS)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--extra", default="", help="Extra raw CLI args to append")


def cmd_record(args: argparse.Namespace) -> None:
    cmd = [
        "lerobot-record",
        *base_robot_args(args),
        f"--teleop.type={args.teleop_type}",
        f"--teleop.left_arm_port={args.left_leader_port}",
        f"--teleop.right_arm_port={args.right_leader_port}",
        f"--dataset.repo_id={args.dataset_repo_id}",
        "--dataset.push_to_hub=false",
        f"--dataset.num_episodes={args.num_episodes}",
        f"--dataset.single_task={args.single_task}",
    ]
    if args.episode_time_s:
        cmd.append(f"--dataset.episode_time_s={args.episode_time_s}")
    if args.reset_time_s:
        cmd.append(f"--dataset.reset_time_s={args.reset_time_s}")
    cmd += shlex.split(args.extra)
    run(cmd, args.dry_run)


def cmd_train(args: argparse.Namespace) -> None:
    cmd = [
        "lerobot-train",
        f"--dataset.repo_id={args.dataset_repo_id}",
        "--policy.type=act",
        f"--output_dir={args.output_dir}",
        f"--job_name={args.job_name}",
        f"--policy.device={args.device}",
    ]
    cmd += shlex.split(args.extra)
    run(cmd, args.dry_run)


def cmd_eval(args: argparse.Namespace) -> None:
    cmd = [
        "lerobot-record",
        *base_robot_args(args),
        f"--dataset.repo_id={args.eval_repo_id}",
        "--dataset.push_to_hub=false",
        f"--dataset.num_episodes={args.num_episodes}",
        f"--dataset.single_task={args.single_task}",
        f"--policy.path={args.policy_path}",
    ]
    cmd += shlex.split(args.extra)
    run(cmd, args.dry_run)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Local ACT workflow for YAMS (no Hub upload)")
    sub = p.add_subparsers(dest="cmd", required=True)

    pr = sub.add_parser("record", help="Record teleop dataset")
    add_common(pr)
    pr.add_argument("--teleop-type", default="bi_yams_leader")
    pr.add_argument("--left-leader-port", default="/dev/ttyACM0")
    pr.add_argument("--right-leader-port", default="/dev/ttyACM1")
    pr.add_argument("--dataset-repo-id", required=True)
    pr.add_argument("--single-task", required=True)
    pr.add_argument("--num-episodes", type=int, default=100)
    pr.add_argument("--episode-time-s", type=int, default=0)
    pr.add_argument("--reset-time-s", type=int, default=0)
    pr.set_defaults(func=cmd_record)

    pt = sub.add_parser("train", help="Train ACT on dataset")
    add_common(pt)
    pt.add_argument("--dataset-repo-id", required=True)
    pt.add_argument("--output-dir", default="outputs/train")
    pt.add_argument("--job-name", default="act_local")
    pt.add_argument("--device", default="cuda")
    pt.set_defaults(func=cmd_train)

    pe = sub.add_parser("eval", help="Run ACT policy and record eval episodes")
    add_common(pe)
    pe.add_argument("--policy-path", required=True)
    pe.add_argument("--eval-repo-id", required=True)
    pe.add_argument("--single-task", required=True)
    pe.add_argument("--num-episodes", type=int, default=20)
    pe.set_defaults(func=cmd_eval)

    return p


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
