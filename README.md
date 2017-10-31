## Distributed training

## Debugging

My debug workflow is

0. Run `python tf_ec2.py clean_launch_and_run`
1. Log into machine (`tf_ec2.py` prints out SSH command)
   shared dir
2. View output in NFS shared directory
3. Make changes on local machine
4. Run `python tf_ec2.py debug`
5. Go back to step 2

