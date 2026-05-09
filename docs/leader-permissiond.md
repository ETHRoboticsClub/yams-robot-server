```bash
echo 'SUBSYSTEM=="tty", ATTRS{idVendor}=="0403", MODE="0666", OWNER="rl"' \
  | sudo tee /etc/udev/rules.d/99-ttyusb.rules
```

Replace `0403` with your actual vendor ID:
```bash
udevadm info -a /dev/ttyUSB0 | grep idVendor | head -1
```

Then reload:
```bash
sudo udevadm control --reload-rules && sudo udevadm trigger
```

Or if you don't want to bother with vendor IDs, blanket rule:
```bash
echo 'SUBSYSTEM=="tty", KERNEL=="ttyUSB*", MODE="0666", OWNER="rl"' \
  | sudo tee /etc/udev/rules.d/99-ttyusb.rules
```

Replug the device after triggering.
