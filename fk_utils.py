import numpy as np
import torch
import re
import torch.nn.functional as F
"""params={}
params['allowed_attrs_file']='./Flipkart/Attribute_allowedvalues.npy'
params['category_attrs_mapping']='./Flipkart/vertical_attributes.npy'
"""
def decode_outputs(params,values_obj,category_mapping,cat_output,val_output):
    """ Remember not to do shuffle and keep the same batch size for both
        Takes as input :
        params: for eval_threshold key
        values_obj: The loaded class Values obj
        category_mapping: The dictionary mapping categories encodings to the strings.
        cat_output: Tensor of size(batch_size,26)
        val_output: Tensor of size(batch_size,802)
        Returns the output after decoding:
        To be passed directly to get_outputs function
    """


    category_list=cat_output.argmax(dim=1)
    category_list=[category_mapping[i.item()] for i in category_list]
    
    m,_=val_output.shape
    val_output_probs=torch.sigmoid(val_output)
#     print(val_output_probs.shape)
    val_output_probs=val_output_probs.tolist()
    return category_list,val_output_probs


# def decode_outputs(threshold,values_obj,category_mapping,cat_output,val_output):
#     """ Remember not to do shuffle and keep the same batch size for both
#         Takes as input :
#         params: for eval_threshold key
#         values_obj: The loaded class Values obj
#         category_mapping: The dictionary mapping categories encodings to the strings.
#         cat_output: Tensor of size(batch_size,26)
#         val_output: Tensor of size(batch_size,802)
#         Returns the output after decoding:
#         To be passed directly to get_outputs function
#     """


#     category_list=cat_output.argmax(dim=1)
#     category_list=[category_mapping[i.item()] for i in category_list]
    
#     val_output_probs=F.sigmoid(val_output)
#     m,_=val_output.shape
#     values_list=[]
    
#     for batch in range(m):
#         ans=list((torch.where(val_output_probs[batch]>threshold)[0]))
#         ans=[values_obj.index2word[i.item()] for i in ans]
#         values_list.append(ans)
    
#     return category_list,values_list

def get_allowed_values(file_name):
    """ Takes as input the file mapping attribute to values returns the dict of all allowed values based on attrs"""
    attibute_allowedvalue_dict = np.load(file_name,allow_pickle=True).tolist()
    return attibute_allowedvalue_dict

def get_allowed_cat_attrs(file_name):
    """ Takes as input the file mapping category to attribute  returns the dict of all allowed attributes based on vat """
    vertical_value_dict = np.load(file_name,allow_pickle=True).tolist()
    return vertical_value_dict

def get_outputs(category_attr_mapping,attr_values_mapping,pred_categories, pred_values,values_obj):
    """ Takes as input the params dict, decoded form of category  and decode form of attribute values ( As a list of items)
        Returns the prediction list as a dict of attrs
        Usage:
        pred_categories=['capri']
        pred_values=[['beach',
          'casual',
          'ethnic','above knee',
          'ankle','sleepwear',
         'sportswear']]

    """
    one_list=set(['top_type','sleeve_length','topwear_length_type','neck_type','suitable_for','pattern_coverage','pattern','ideal_for','sleeve_style','ornamentation_type','vertical','bottomwear_length_type','heel_pattern','closure_type','hem','dupatta_included','style_type','character','short_type','sari_type','detail_placement','belt_included','trouser_type','surface_styling','sleeve_details','border_details','border_length','pleats','footwear_pattern_placement','bottom_type','slipper_flip_flop_type','distressed','rise','type_of_embroidery','faded','skirt_type','uniform','embellished','jegging_type','sandal_type'])
    n_items=len(pred_categories)
    #category_attr_mapping=get_allowed_cat_attrs(params['category_attrs_mapping'])
    #attr_values_mapping=get_allowed_values(params['allowed_attrs_file'])
    predictions=[]
#     if 'vertical' in category_attr_mapping.keys():
#         del category_attr_mapping['vertical']
    for key in category_attr_mapping.keys():
        category_attr_mapping[key]=[i for i in category_attr_mapping[key] if i!='vertical']
    
#     print(values_obj.word2index.keys())
#     print(pred_values)
    
    
#     print(category_attr_mapping)
    for key in attr_values_mapping.keys():
        attr_values_mapping[key]=set(attr_values_mapping[key])
    
    score_lists=[]
    for i in range(n_items):
        score_dict={}
        for attr in category_attr_mapping[pred_categories[i]]:
            allowed_vals=attr_values_mapping[attr]
#             print(attr)
            templist=[]
            for k in allowed_vals:
                k=re.sub(r'[^\x00-\x7F]+', ' ', k)
#                 print(k)
                templist.append((pred_values[i][values_obj.word2index[k]],k))
            score_dict[attr]=templist
        score_lists.append(score_dict)
    
    pred_list=[]
    
    for i in range(n_items):
        pred_dict={}
        for key,val in score_lists[i].items():
            val=sorted(val, key=lambda x: x[0], reverse=True)
            if key in one_list:
                val=list(map(lambda x: x[1], val[:1]))
            else:
                val=list(map(lambda x: x[1], val[:2]))
            pred_dict[key]=val
        pred_list.append(pred_dict)
    return pred_list

# def get_outputs(category_attr_mapping,attr_values_mapping,pred_categories, pred_values):
#     """ Takes as input the params dict, decoded form of category  and decode form of attribute values ( As a list of items)
#         Returns the prediction list as a dict of attrs
#         Usage:
#         pred_categories=['capri']
#         pred_values=[['beach',
#           'casual',
#           'ethnic','above knee',
#           'ankle','sleepwear',
#          'sportswear']]

#     """
#     n_items=len(pred_categories)
#     #category_attr_mapping=get_allowed_cat_attrs(params['category_attrs_mapping'])
#     #attr_values_mapping=get_allowed_values(params['allowed_attrs_file'])
#     predictions=[]
# #     for key in category_attr_mapping.keys():
# #         category_attr_mapping[key]=[i for i in category_attr_mapping[key] if key!='vertical']
    
#     for key in attr_values_mapping.keys():
#         attr_values_mapping[key]=set(attr_values_mapping[key])
        
        
#     for i in range(n_items):
#         tempdict={}
#         preds=set(pred_values[i])
        
#         for attr in category_attr_mapping[pred_categories[i]]:
#             allowed_vals_cats=attr_values_mapping[attr]
#             tempdict[attr]=list(allowed_vals_cats.intersection(preds))
        
#         predictions.append(tempdict)
#     return predictions



def get_iou(*dicts):
    """Takes input as ground truth dict and predicted attributes dict """
    comm_keys = dicts[0].keys()
    for d in dicts[1:]:
        # intersect keys first
        comm_keys &= d.keys()
    # then build a result dict with nested comprehension
    score=[]
    for key in comm_keys:
        gt=set(dicts[0][key])
        pred=set(dicts[1][key])
        union=gt.union(pred)
        inter=gt.intersection(pred)
        score.append(len(inter)/len(union))
#     To average the score dividing by the ground truth length
    return sum(score)/len(dicts[0])

# For calculating the score
def evaluate_model_scores(ground_truth,predictions):
    """
    The function takes as input two list of dictionaries where each dictionary 
    contains keys as attributes and values as the attribute values
    Returns average scores
    Usage:
    dict1={'a':[2,3],'b':[6,7] }
    dict2={'c':[7],'a':[2,3],'b':[9] }
    gt=[]
    pred=[]
    gt.append(dict1)
    pred.append(dict2)
    gt.append(dict2)
    pred.append(dict1)

    evaluate_model_scores(gt,pred)
    Returns the average score
    """
    n_items=len(ground_truth)
    total_scores=[]
    for i in range(n_items):
        total_scores.append( get_iou(ground_truth[i],predictions[i]) )
#     print(total_scores)
    return sum(total_scores)/len(total_scores)

