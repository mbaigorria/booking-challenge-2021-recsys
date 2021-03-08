#!/bin/sh

URL=https://shoutout.wix.com/so/c4NRCVgeV/c?w=OUERqDAyWlD4w8xMugq00JMzsLO6ZoB9fQcWfVv409k.eyJ1IjoiaHR0cHM6Ly8wMzUxMDVmNy1hZTMyLTQ3YjYtYTI1Yi04N2FmNzkyNGM3ZWEudXNyZmlsZXMuY29tL2FyY2hpdmVzLzMwOTFiZV84Mzc4NGYzMzA1NWM0NzkxYWUyYjY4ODdlMTFkZGYwOS56aXA_ZG49QktOR3Rlc3RkYXRhV1NETTIwMjEuemlwIiwiciI6IjAzMzQzYjU3LTQ5MGYtNDk3OS02NGFjLTZjZTJlYjBiZDU0NyIsIm0iOiJtYWlsIiwiYyI6IjRlNjU2ZjFhLTEzOTgtNDRiNy04OWJiLTRhNjc5Mjc1MGExYSJ9
if [ ! -f "resources/booking_train_set.csv" ]; then
  wget $URL -O dataset.zip
  unzip dataset.zip -d resources
  rm dataset.zip
fi