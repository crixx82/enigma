
# ENIGMA Consortium - Joint-Variation Graph Analysis

This project was developed to analyze structural MRI data of the ENIGMA consortium. This script uses previously extracted morphological data to compute individualized joint-variantion graphs of cortical thickness, extracts a set of topological properties of such graphs, and computes descriptive statistics for meta-analysis. 
Specifically, this script was created to investigate alterations in patterns of cortical thickness in individuals with anorexia nervosa.


## Work environment setup

The project use a preconfig environment setupped in a Docker solution. You need to install the follow dependences for Docker environment in accord with your OS System.

For Windows 10 and 11 you need to enable at BIOS level hardware virtualization support and follow the WSL 2 installation flow on the official Docker guide.

For Linux System you need to have a 64-bit kernel and CPU support for virtualization, KVM virtualization support and QEMU al least be version 5.2.

Official Docker guide.
https://docs.docker.com/get-docker/


## Getting Started

From a Linux subsystem shall clone the git project on the your home directory.

```bash
  git clone https://github.com/crixx82/enigma.git
```
If you are not confidential with git flow, you can dawnload the source code in zip format in thr upper right side of the windows. (https://github.com/crixx82/enigma/archive/refs/heads/main.zip).
Then you need to unzip it in a folder named enigma.
Find the `docker-compose.yaml` inside enigma and change the placeholder username with your subsystem username.
```yaml
.
.
  panel:
    build:
      args:
        user: {YOUR-USER-NAME}
        uid: 5000
      context: ./
      dockerfile: Dockerfile
.
.
```

Before run Docker, you need to gant privilege to the output project folder. Please keep attention at your project folder location. Now you are aready to build the Docker image.

```bash
  cd /home/{YOUR-USER-NAME}/{OPTIONAL-YOUR-GIT-FOLDER}/enigma
  chmod 777 output/
  chmod 777 output/log.txt
  docker compose up
```
**MacOs users: if the Docker image build process fail tray to delete Docker's configuration.
```bash
  rm ~/.docker/config.json
```

At the end of the image build process, the analisis script will run and you can find the output on the output project directory:

```bash
  cd /home/YOUR-USER-NAME/enigma/output
```
## Support

For support, please contact email or join our Slack channel.


## License

[MIT](https://choosealicense.com/licenses/mit/)

