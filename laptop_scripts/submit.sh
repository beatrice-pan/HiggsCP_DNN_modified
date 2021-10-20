python ../main.py -e 25 -t nn_rhorho -i $RHORHO_DATA -f Variant-All --training_method regr_argmaxs
python ../main.py -e 25 -t nn_rhorho -i $RHORHO_DATA -f Variant-All --training_method regr_c012s
python ../main.py -e 25 -t nn_rhorho -i $RHORHO_DATA -f Variant-All --training_method regr_weights --num_classes 21

python ../main.py -e 25 -t nn_rhorho -i $RHORHO_DATA -f Variant-All --training_method soft_argmaxs --num_classes 21
python ../main.py -e 25 -t nn_rhorho -i $RHORHO_DATA -f Variant-All --training_method soft_c012s --hits_c012s hits_c0s --num_classes 21
python ../main.py -e 25 -t nn_rhorho -i $RHORHO_DATA -f Variant-All --training_method soft_c012s --hits_c012s hits_c1s --num_classes 21
python ../main.py -e 25 -t nn_rhorho -i $RHORHO_DATA -f Variant-All --training_method soft_c012s --hits_c012s hits_c2s --num_classes 21
python ../main.py -e 25 -t nn_rhorho -i $RHORHO_DATA -f Variant-All --num_classes 21
