# TPGSR-1
python3 main.py --arch="tsrn_tl_cascade" --batch_size=256 --STN --mask --use_distill --gradient --sr_share --stu_iter=1 --vis_dir='vis_TPGSR-TSRN-ASTER' --rec="aster" --test_model="ASTER"

python3 main.py --arch="tsrn_tl_cascade" --batch_size=256 --STN --mask --use_distill --gradient --sr_share --stu_iter=1 --vis_dir='vis_TPGSR-TSRN-MORAN' --rec="moran" --test_model="MORAN"

python3 main.py --arch="tsrn_tl_cascade" --batch_size=512 --STN --mask --use_distill --gradient --sr_share --stu_iter=1 --vis_dir='vis_TPGSR-TSRN-CRNN' --rec="crnn" --test_model="CRNN"



# TPGSR-3
python3 main.py --arch="tsrn_tl_cascade" --batch_size=256 --STN --mask --use_distill --gradient --sr_share --stu_iter=3 --vis_dir='vis_TPGSR-TSRN-ASTER-3' --rec="aster" --test_model="ASTER" 

python3 main.py --arch="tsrn_tl_cascade" --batch_size=256 --STN --mask --use_distill --gradient --sr_share --stu_iter=3 --vis_dir='vis_TPGSR-TSRN-MORAN-3' --rec="moran" --test_model="MORAN"

python3 main.py --arch="tsrn_tl_cascade" --batch_size=200 --STN --mask --use_distill --gradient --sr_share --stu_iter=3 --vis_dir='vis_TPGSR-TSRN-CRNN-3' --rec="crnn" --test_model="CRNN" --sr_share



# DEBUG
python3 main.py --arch="tsrn_tl_cascade" --batch_size=256 --STN --mask --use_distill --gradient --sr_share --stu_iter=3 --vis_dir='DEBUG-ASTER-3' --rec="aster" --test_model="ASTER" 

python3 main.py --arch="tsrn_tl_cascade" --batch_size=256 --STN --mask --use_distill --gradient --sr_share --stu_iter=3 --vis_dir='DEBUG-MORAN-3' --rec="moran" --test_model="MORAN"

python3 main.py --arch="tsrn_tl_cascade" --batch_size=160 --STN --mask --use_distill --gradient --sr_share --stu_iter=3 --vis_dir='DEBUG-CRNN-3' --rec="crnn" --test_model="CRNN" 

python3 main.py --arch="bicubic" --batch_size=256 --STN --mask --use_distill --gradient --sr_share --stu_iter=3 --vis_dir='Bicubic' --rec="aster" --test_model="ASTER"
