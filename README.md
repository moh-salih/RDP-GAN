# RDP-GAN 

** train without privacy **  
`py train.py --epochs=300 --batch_size=64 --dataset=mnist save_interval=1 --privacy_mode=no_privacy`  

** train without add to loss technique privacy **  
`py train.py --epochs=300 --batch_size=64 --dataset=mnist save_interval=1 --privacy_mode=add_to_loss`  

** train without add to weights technique privacy **  
`py train.py --epochs=300 --batch_size=64 --dataset=mnist save_interval=1 --privacy_mode=add_to_loss`  



