python3 main.py --arch="tsrn_tl_cascade" --test_model="ASTER" --batch_size=256 --STN --mask  --sr_share --gradient --go_test --stu_iter=1 --vis_dir='default' --resume="./ckpt/vis_TPGSR-TSRN-ASTER/model_best_aster.pth" 


python3 main.py --arch="tsrn_tl_cascade" --test_model="MORAN" --batch_size=256 --STN --mask  --sr_share --gradient --go_test --stu_iter=1 --vis_dir='default' --resume="./ckpt/vis_TPGSR-TSRN-MORAN/model_best_moran.pth" 


python3 main.py --arch="tsrn_tl_cascade" --test_model="CRNN" --batch_size=512 --STN --mask  --sr_share --gradient --go_test --stu_iter=1 --vis_dir='default' --resume="./ckpt/vis_TPGSR-TSRN-CRNN/model_best_crnn.pth" 






python3 main.py --arch="tsrn_tl_cascade" --test_model="CRNN" --batch_size=512 --STN --mask  --sr_share --gradient --go_test --stu_iter=3 --vis_dir='default' --resume="./ckpt/CRNN-3-B-ECA-TEA/model_best_crnn.pth" 
