import os
from pathlib import Path

DEFAULT_SSH_DIRECTORY = Path(os.path.expanduser("~") + "/.ssh")

# Setup Github connection
def create_github_sshkey_on_machine(
    github_email: str, ssh_directory: Path = DEFAULT_SSH_DIRECTORY
):
    github_ssh_config = """
    Host *.github.com
    AddKeysToAgent yes
    IdentityFile ~/.ssh/id_ed25519
    """.strip()

    # if ssh_directory:
    #     ssh_directory = ssh_directory.replace("~", os.path.expanduser("~"))

    os.system(f"touch {ssh_directory}/id_ed25519")
    os.system(
        f'ssh-keygen -q -t ed25519 -N "" -f {ssh_directory}/id_ed25519 -C "{github_email}" <<< y'
    )
    os.system(f"chmod 700 {ssh_directory}/id_ed25519")
    os.system('eval "$(ssh-agent -s)"')
    os.system(f"touch {ssh_directory}/config")

    with open(f"{ssh_directory}/config", "r") as f:
        ssh_config = f.read()
        if github_ssh_config not in ssh_config:
            os.system(f"echo '{github_ssh_config}' >> ~/.ssh/config")
    os.system("ssh-add ~/.ssh/id_ed25519")
    os.system(f"cat {ssh_directory}/config")
