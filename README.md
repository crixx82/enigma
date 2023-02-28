
# Enigma

Please provide a project description.




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
Change the user name on the `docker-compose.yaml` with your subsystem username.
```yaml
.
.
  panel:
    build:
      args:
        user: YOUR-USER-NAME
        uid: 1001
      context: ./
      dockerfile: Dockerfile
.
.
```

Now you are aready to build the Docker image.

```bash
  cd /home/YOUR-USER-NAME/enigma
  docker compose up
```

At the end of the image build process, the analisis script will run and you can find the output on the output project directory:

```bash
  cd /home/YOUR-USER-NAME/enigma/data/output
```
## Support

For support, please contact email or join our Slack channel.


## License

[MIT](https://choosealicense.com/licenses/mit/)

