python ../main.py -e 5 -t nn_rhorho -i $RHORHO_DATA -f Variant-All --training_method soft_c012s --hits_c012s hits_c0s --num_classes 21
python ../main.py -e 5 -t nn_rhorho -i $RHORHO_DATA -f Variant-All --training_method soft_c012s --hits_c012s hits_c1s --num_classes 21
python ../main.py -e 5 -t nn_rhorho -i $RHORHO_DATA -f Variant-All --training_method soft_c012s --hits_c012s hits_c2s --num_classes 21

python ../main.py -e 5 -t nn_rhorho -i $RHORHO_DATA -f Variant-All --training_method soft_c012s --hits_c012s hits_c0s --num_classes 51
python ../main.py -e 5 -t nn_rhorho -i $RHORHO_DATA -f Variant-All --training_method soft_c012s --hits_c012s hits_c1s --num_classes 51
python ../main.py -e 5 -t nn_rhorho -i $RHORHO_DATA -f Variant-All --training_method soft_c012s --hits_c012s hits_c2s --num_classes 51

