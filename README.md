## This project focuses on deploying a simple ElasticNet on AWS with S3 bucket for data storage and EC2 instance for virtual server.

### using MLFLOW on AWS
1. Create IAM user with AdminstratorAccess
2. Put credentials in aws configure
3. Create S3 bucket
4. Create EC2 machine for Ubuntu and add security groups 5000 port

### inside ec2 instance terminal:
1. sudo apt update
2. sudo apt install python3-pip
3. sudo apt install pipenv
4. sudo apt install virtualenv
5. mkdir mlflow
6. cd mlflow
7. pipenv install mlflow
8. pipenv install awscli
9. pipenv install boto3
10. pipenv shell

11. aws configure

12. mlflow server --host 0.0.0.0 --default-artifact-root s3://mlflowtracking52 --workers 1

### open public ipv4 DNS to the port 5000

### set uri in your local terminal and in your code
export MLFLOW_TRACKING_URI=http://ec2-13-51-56-3.eu-north-1.compute.amazonaws.com:5000/ (not live)