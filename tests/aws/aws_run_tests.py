import time
import json
import subprocess
from pathlib import Path

# Global EC2 choices outside the account
EC2_AMI = "ami-0e720ce540d437165"  # Deep Learning Base Proprietary Nvidia Driver GPU AMI (Ubuntu 20.04)
EC2_INSTANCE_TYPE = "g4dn.xlarge"
EC2_REGION = "us-east-1"
EC2_USER = "ubuntu"

# Account-specific entities and prerequisites
EC2_KEY_NAME = "first"  # make a key pair, import into your ssh manager
EC2_SECURITY_GROUP_ID = "sg-065cc73ca303ec0bc"  # make a security group with allowed inbound ssh access (port 22)

# Extra helper variables
LPATH_PROJECT = str(Path(__file__).absolute().parent.parent.parent)
RPATH_PROJECT = f"/home/{EC2_USER}/torch-fidelity"
INSTANCE_NAME = "torch-fidelity"
CMD_SSH_CUSTOM = [
    # fmt: off
    "ssh", "-q",
    "-o", "StrictHostKeyChecking=no",
    "-o", "LogLevel=ERROR",
    "-o", "ConnectTimeout=180",
    # fmt: on
]


def join_cmd(cmd_list):
    return " ".join([a if " " not in a else f'"{a}"' for a in cmd_list])


def run_command_retry(cmd, timeout_sec=5):
    while True:
        try:
            completed_process = subprocess.run(cmd, shell=False, check=True, text=True, capture_output=True)
            print(completed_process.stderr.strip())
            return completed_process.stdout.strip()
        except subprocess.CalledProcessError:
            print(f"Retrying in {timeout_sec} seconds...")
            time.sleep(timeout_sec)


def main():
    # fmt: off
    instance_id = run_command_retry([
        "aws", "ec2", "describe-instances",
        "--filters", f"Name=tag:Name,Values={INSTANCE_NAME}",
                     f"Name=instance-state-name,Values=pending,running,stopping,stopped",
        "--query", "Reservations[*].Instances[*].InstanceId",
        "--output", "text",
    ])
    # fmt: on

    if instance_id == "":
        # https://docs.aws.amazon.com/cli/latest/reference/ec2/run-instances.html
        print("Launching instance...")
        instance_tag = "ResourceType=instance,Tags=[{Key=Name,Value=" + INSTANCE_NAME + "}]"
        cmd_launch = [
            # fmt: off
            "aws", "ec2", "run-instances",
            "--tag-specifications", instance_tag,
            "--instance-type", EC2_INSTANCE_TYPE,
            "--image-id", EC2_AMI,
            "--key-name", EC2_KEY_NAME,
            "--security-group-ids", EC2_SECURITY_GROUP_ID,
            "--block-device-mappings", 'DeviceName="/dev/sda1",Ebs={VolumeSize=200}',
            # fmt: on
        ]
        print(f"CMD: {join_cmd(cmd_launch)}")
        response = json.loads(run_command_retry(cmd_launch, 60))
        instance_id = response["Instances"][0]["InstanceId"]
    else:
        print("Starting instance...")
        # fmt: off
        run_command_retry([
            "aws", "ec2", "start-instances",
            "--instance-ids", instance_id
        ])
        # fmt: on
    print(f"InstanceID: {instance_id}")

    # fmt: off
    dns_response = json.loads(run_command_retry([
        "aws", "ec2", "describe-instances",
        "--region", EC2_REGION,
        "--instance-ids", instance_id,
    ]))
    # fmt: on
    instance_dns = dns_response["Reservations"][0]["Instances"][0]["PublicDnsName"]
    print(f"InstanceDNS: {instance_dns}")

    print("Copying files to instance...")
    cmd_ssh_custom_joined = join_cmd(CMD_SSH_CUSTOM)
    cmd_rsync = [
        # fmt: off
        "rsync", "--relative", "-av",
        "-e", cmd_ssh_custom_joined,
        f"{LPATH_PROJECT}/./",
        f"{EC2_USER}@{instance_dns}:{RPATH_PROJECT}/",
        # fmt: on
    ]
    print(f"CMD: {join_cmd(cmd_rsync)}")
    run_command_retry(cmd_rsync)

    print("Starting testing in a tmux session...")
    cmd_ssh = CMD_SSH_CUSTOM + [f"{EC2_USER}@{instance_dns}"]
    print(f"CMD: {join_cmd(cmd_ssh)}")
    run_command_retry(cmd_ssh + ["bash", f"{RPATH_PROJECT}/tests/aws/aws_tmux_wrapper.sh"])
    print(f"CMD: {join_cmd(cmd_ssh)} -t tmux attach-session -t fid")


if __name__ == "__main__":
    main()
