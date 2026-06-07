grammar = {

    # ── WordNet Taxonomic ──────────────────────────────────────────────────────

    "hypernym": """
        VP -> TRANS [0.5]
        VP -> COP [0.5]
        TRANS -> V [1.0]
        V -> 'exemplifies' [0.25]
        V -> 'specializes' [0.25]
        V -> 'refines' [0.25]
        V -> 'extends' [0.25]
        COP -> AUX ADJP PP [1.0]
        AUX -> 'is' [0.34]
        AUX -> 'remains' [0.33]
        AUX -> 'appears' [0.33]
        ADJP -> 'subordinate' [0.34]
        ADJP -> 'more_specific' [0.33]
        ADJP -> 'subsumed' [0.33]
        PP -> 'to' [0.5]
        PP -> 'under' [0.5]

        # Expansion
        VP_EXP_hyponym -> TRANS_HYPONYM 'tgt' CONJ VP 'tgt_2' [0.5]
        VP_EXP_hyponym -> COP_HYPONYM 'tgt' CONJ VP 'tgt_2' [0.5]
        
        CONJ -> 'and' [1.0]
        
        TRANS_HYPONYM -> V_HYPO [1.0]
        V_HYPO -> 'generalizes' [0.25] | 'encompasses' [0.25] | 'subsumes' [0.25] | 'categorizes' [0.25]
        COP_HYPONYM -> AUX_HYPO ADJP_HYPO PP_HYPO [1.0]
        AUX_HYPO -> 'is' [0.34] | 'remains' [0.33] | 'appears' [0.33]
        ADJP_HYPO -> 'broader' [0.34] | 'superordinate' [0.33] | 'more_general' [0.33]
        PP_HYPO -> 'than' [0.5] | 'over' [0.5]
    """,

    "hyponym": """
        VP -> TRANS [0.5]
        VP -> COP [0.5]
        TRANS -> V [1.0]
        V -> 'generalizes' [0.25]
        V -> 'encompasses' [0.25]
        V -> 'subsumes' [0.25]
        V -> 'categorizes' [0.25]
        COP -> AUX ADJP PP [1.0]
        AUX -> 'is' [0.34]
        AUX -> 'remains' [0.33]
        AUX -> 'appears' [0.33]
        ADJP -> 'broader' [0.34]
        ADJP -> 'superordinate' [0.33]
        ADJP -> 'more_general' [0.33]
        PP -> 'than' [0.5]
        PP -> 'over' [0.5]

        # Expansion
        VP_EXP_hypernym -> TRANS_HYPERNYM 'tgt' CONJ VP 'tgt_2' [0.5]
        VP_EXP_hypernym -> COP_HYPERNYM 'tgt' CONJ VP 'tgt_2' [0.5]
        
        CONJ -> 'and' [1.0]
        
        TRANS_HYPERNYM -> V_HYP [1.0]
        V_HYP -> 'exemplifies' [0.25] | 'specializes' [0.25] | 'refines' [0.25] | 'extends' [0.25]
        COP_HYPERNYM -> AUX_HYP ADJP_HYP PP_HYP [1.0]
        AUX_HYP -> 'is' [0.34] | 'remains' [0.33] | 'appears' [0.33]
        ADJP_HYP -> 'subordinate' [0.34] | 'more_specific' [0.33] | 'subsumed' [0.33]
        PP_HYP -> 'to' [0.5] | 'under' [0.5]
    """,

    "instance_hypernym": """
        VP -> TRANS [0.5]
        VP -> COP [0.5]
        TRANS -> V [1.0]
        V -> 'instantiates' [0.34]
        V -> 'exemplifies' [0.33]
        V -> 'represents' [0.33]
        COP -> AUX ADJP PP [1.0]
        AUX -> 'is' [0.34]
        AUX -> 'remains' [0.33]
        AUX -> 'appears' [0.33]
        ADJP -> 'classified' [0.34]
        ADJP -> 'categorized' [0.33]
        ADJP -> 'recognized' [0.33]
        PP -> 'as' [0.5]
        PP -> 'under' [0.5]

        # Expansion
        VP_EXP_instance_hyponym -> TRANS_INSTANCE_HYPONYM 'tgt' CONJ COP_INSTANCE_HYPONYM 'tgt_2' [0.5]
        VP_EXP_instance_hyponym -> COP_INSTANCE_HYPONYM_BASE 'tgt' CONJ TRANS_INSTANCE_HYPONYM 'tgt_2' [0.5]
        
        CONJ -> 'and' [1.0]
        
        TRANS_INSTANCE_HYPONYM -> V_I_HYPO [1.0]
        V_I_HYPO -> 'encompasses' [0.34] | 'includes' [0.33] | 'admits' [0.33]
        
        COP_INSTANCE_HYPONYM_BASE -> AUX ADJP PP [1.0]
        COP_INSTANCE_HYPONYM -> AUX_I_HYPO ADJP_I_HYPO PP_I_HYPO [1.0]
        AUX_I_HYPO -> 'is' [0.34] | 'remains' [0.33] | 'appears' [0.33]
        ADJP_I_HYPO -> 'exemplified' [0.34] | 'represented' [0.33] | 'instantiated' [0.33]
        PP_I_HYPO -> 'by' [0.5] | 'through' [0.5]
    """,

    "instance_hyponym": """
        VP -> TRANS [0.5]
        VP -> COP [0.5]
        TRANS -> V [1.0]
        V -> 'encompasses' [0.34]
        V -> 'includes' [0.33]
        V -> 'admits' [0.33]
        COP -> AUX ADJP PP [1.0]
        AUX -> 'is' [0.34]
        AUX -> 'remains' [0.33]
        AUX -> 'appears' [0.33]
        ADJP -> 'exemplified' [0.34]
        ADJP -> 'represented' [0.33]
        ADJP -> 'instantiated' [0.33]
        PP -> 'by' [0.5]
        PP -> 'through' [0.5]

        # Expansion
        VP_EXP_instance_hypernym -> TRANS_INSTANCE_HYPERNYM 'tgt' CONJ COP_INSTANCE_HYPERNYM 'tgt_2' [0.5]
        VP_EXP_instance_hypernym -> COP_INSTANCE_HYPERNYM_BASE 'tgt' CONJ TRANS_INSTANCE_HYPERNYM 'tgt_2' [0.5]
        
        CONJ -> 'and' [1.0]
        
        TRANS_INSTANCE_HYPERNYM -> V_I_HYP [1.0]
        V_I_HYP -> 'instantiates' [0.34] | 'exemplifies' [0.33] | 'represents' [0.33]
        
        COP_INSTANCE_HYPERNYM_BASE -> AUX ADJP PP [1.0]
        COP_INSTANCE_HYPERNYM -> AUX_I_HYP ADJP_I_HYP PP_I_HYP [1.0]
        AUX_I_HYP -> 'is' [0.34] | 'remains' [0.33] | 'appears' [0.33]
        ADJP_I_HYP -> 'classified' [0.34] | 'categorized' [0.33] | 'recognized' [0.33]
        PP_I_HYP -> 'as' [0.5] | 'under' [0.5]
    """,

    # ── WordNet Part-Whole ─────────────────────────────────────────────────────

    "part_meronym": """
        VP -> TRANS [0.5]
        VP -> COP [0.5]
        TRANS -> V [1.0]
        V -> 'contains' [0.25]
        V -> 'includes' [0.25]
        V -> 'harbors' [0.25]
        V -> 'features' [0.25]
        COP -> AUX ADJP PP [1.0]
        AUX -> 'is' [0.34]
        AUX -> 'remains' [0.33]
        AUX -> 'appears' [0.33]
        ADJP -> 'composed' [0.34]
        ADJP -> 'comprised' [0.33]
        ADJP -> 'made_up' [0.33]
        PP -> 'of' [1.0]

        # Expansion
        VP_EXP_part_holonym -> TRANS_PART_HOLONYM 'tgt' CONJ COP_PART_HOLONYM 'tgt_2' [0.5]
        VP_EXP_part_holonym -> COP_PART_HOLONYM_BASE 'tgt' CONJ TRANS_PART_HOLONYM 'tgt_2' [0.5]
        
        CONJ -> 'and' [1.0]
        
        TRANS_PART_HOLONYM -> V_P_HOLO [1.0]
        V_P_HOLO -> 'completes' [0.34] | 'constitutes' [0.33] | 'supplements' [0.33]
        
        COP_PART_HOLONYM_BASE -> AUX ADJP PP [1.0]
        COP_PART_HOLONYM -> AUX_P_HOLO ADJP_P_HOLO PP_P_HOLO [1.0]
        AUX_P_HOLO -> 'is' [0.34] | 'remains' [0.33] | 'appears' [0.33]
        ADJP_P_HOLO -> 'contained' [0.34] | 'embedded' [0.33] | 'integrated' [0.33]
        PP_P_HOLO -> 'in' [0.34] | 'within' [0.33] | 'into' [0.33]
    """,

    "part_holonym": """
        VP -> TRANS [0.5]
        VP -> COP [0.5]
        TRANS -> V [1.0]
        V -> 'completes' [0.34]
        V -> 'constitutes' [0.33]
        V -> 'supplements' [0.33]
        COP -> AUX ADJP PP [1.0]
        AUX -> 'is' [0.34]
        AUX -> 'remains' [0.33]
        AUX -> 'appears' [0.33]
        ADJP -> 'contained' [0.34]
        ADJP -> 'embedded' [0.33]
        ADJP -> 'integrated' [0.33]
        PP -> 'in' [0.34]
        PP -> 'within' [0.33]
        PP -> 'into' [0.33]

        # Expansion
        VP_EXP_part_meronym -> TRANS_PART_MERONYM 'tgt' CONJ COP_PART_MERONYM 'tgt_2' [0.5]
        VP_EXP_part_meronym -> COP_PART_MERONYM_BASE 'tgt' CONJ TRANS_PART_MERONYM 'tgt_2' [0.5]
        
        CONJ -> 'and' [1.0]
        
        TRANS_PART_MERONYM -> V_P_MERO [1.0]
        V_P_MERO -> 'contains' [0.25] | 'includes' [0.25] | 'harbors' [0.25] | 'features' [0.25]
        
        COP_PART_MERONYM_BASE -> AUX ADJP PP [1.0]
        COP_PART_MERONYM -> AUX_P_MERO ADJP_P_MERO PP_P_MERO [1.0]
        AUX_P_MERO -> 'is' [0.34] | 'remains' [0.33] | 'appears' [0.33]
        ADJP_P_MERO -> 'composed' [0.34] | 'comprised' [0.33] | 'made_up' [0.33]
        PP_P_MERO -> 'of' [1.0]
    """,

    "member_meronym": """
        VP -> TRANS [0.5]
        VP -> COP [0.5]
        TRANS -> V [1.0]
        V -> 'includes' [0.25]
        V -> 'admits' [0.25]
        V -> 'encompasses' [0.25]
        V -> 'counts' [0.25]
        COP -> AUX ADJP PP [1.0]
        AUX -> 'is' [0.34]
        AUX -> 'remains' [0.33]
        AUX -> 'appears' [0.33]
        ADJP -> 'composed' [0.34]
        ADJP -> 'inclusive' [0.33]
        ADJP -> 'populated' [0.33]
        PP -> 'of' [0.5]
        PP -> 'with' [0.5]

        # Expansion
        VP_EXP_member_holonym -> TRANS_MEMBER_HOLONYM 'tgt' CONJ COP_MEMBER_HOLONYM 'tgt_2' [0.5]
        VP_EXP_member_holonym -> COP_MEMBER_HOLONYM_BASE 'tgt' CONJ TRANS_MEMBER_HOLONYM 'tgt_2' [0.5]
        
        CONJ -> 'and' [1.0]
        
        TRANS_MEMBER_HOLONYM -> V_M_HOLO [1.0]
        V_M_HOLO -> 'joins' [0.34] | 'enters' [0.33] | 'populates' [0.33]
        
        COP_MEMBER_HOLONYM_BASE -> AUX ADJP PP [1.0]
        COP_MEMBER_HOLONYM -> AUX_M_HOLO ADJP_M_HOLO PP_M_HOLO [1.0]
        AUX_M_HOLO -> 'is' [0.34] | 'remains' [0.33] | 'appears' [0.33]
        ADJP_M_HOLO -> 'affiliated' [0.34] | 'associated' [0.33] | 'enrolled' [0.33]
        PP_M_HOLO -> 'with' [0.5] | 'in' [0.5]
    """,

    "member_holonym": """
        VP -> TRANS [0.5]
        VP -> COP [0.5]
        TRANS -> V [1.0]
        V -> 'joins' [0.34]
        V -> 'enters' [0.33]
        V -> 'populates' [0.33]
        COP -> AUX ADJP PP [1.0]
        AUX -> 'is' [0.34]
        AUX -> 'remains' [0.33]
        AUX -> 'appears' [0.33]
        ADJP -> 'affiliated' [0.34]
        ADJP -> 'associated' [0.33]
        ADJP -> 'enrolled' [0.33]
        PP -> 'with' [0.5]
        PP -> 'in' [0.5]

        # Expansion
        VP_EXP_member_meronym -> TRANS_MEMBER_MERONYM 'tgt' CONJ COP_MEMBER_MERONYM 'tgt_2' [0.5]
        VP_EXP_member_meronym -> COP_MEMBER_MERONYM_BASE 'tgt' CONJ TRANS_MEMBER_MERONYM 'tgt_2' [0.5]
        
        CONJ -> 'and' [1.0]
        
        TRANS_MEMBER_MERONYM -> V_M_MERO [1.0]
        V_M_MERO -> 'includes' [0.25] | 'admits' [0.25] | 'encompasses' [0.25] | 'counts' [0.25]
        
        COP_MEMBER_MERONYM_BASE -> AUX ADJP PP [1.0]
        COP_MEMBER_MERONYM -> AUX_M_MERO ADJP_M_MERO PP_M_MERO [1.0]
        AUX_M_MERO -> 'is' [0.34] | 'remains' [0.33] | 'appears' [0.33]
        ADJP_M_MERO -> 'composed' [0.34] | 'inclusive' [0.33] | 'populated' [0.33]
        PP_M_MERO -> 'of' [0.5] | 'with' [0.5]
    """,

    "substance_meronym": """
        VP -> TRANS [0.5]
        VP -> COP [0.5]
        TRANS -> V [1.0]
        V -> 'contains' [0.34]
        V -> 'incorporates' [0.33]
        V -> 'embeds' [0.33]
        COP -> AUX ADJP PP [1.0]
        AUX -> 'is' [0.34]
        AUX -> 'remains' [0.33]
        AUX -> 'appears' [0.33]
        ADJP -> 'composed' [0.34]
        ADJP -> 'comprised' [0.33]hascontex
        ADJP -> 'constituted' [0.33]
        PP -> 'of' [1.0]

        # Expansion
        VP_EXP_substance_holonym -> TRANS_SUBSTANCE_HOLONYM 'tgt' CONJ COP_SUBSTANCE_HOLONYM 'tgt_2' [0.5]
        VP_EXP_substance_holonym -> COP_SUBSTANCE_HOLONYM_BASE 'tgt' CONJ TRANS_SUBSTANCE_HOLONYM 'tgt_2' [0.5]
        
        CONJ -> 'and' [1.0]
        
        TRANS_SUBSTANCE_HOLONYM -> V_S_HOLO [1.0]
        V_S_HOLO -> 'constitutes' [0.34] | 'forms' [0.33] | 'composes' [0.33]
        
        COP_SUBSTANCE_HOLONYM_BASE -> AUX ADJP PP [1.0]
        COP_SUBSTANCE_HOLONYM -> AUX_S_HOLO ADJP_S_HOLO PP_S_HOLO [1.0]
        AUX_S_HOLO -> 'is' [0.34] | 'remains' [0.33] | 'appears' [0.33]
        ADJP_S_HOLO -> 'integral' [0.34] | 'inherent' [0.33] | 'essential' [0.33]
        PP_S_HOLO -> 'to' [0.5] | 'in' [0.5]
    """,

    "substance_holonym": """
        VP -> TRANS [0.5]
        VP -> COP [0.5]
        TRANS -> V [1.0]
        V -> 'constitutes' [0.34]
        V -> 'forms' [0.33]
        V -> 'composes' [0.33]
        COP -> AUX ADJP PP [1.0]
        AUX -> 'is' [0.34]
        AUX -> 'remains' [0.33]
        AUX -> 'appears' [0.33]
        ADJP -> 'integral' [0.34]
        ADJP -> 'inherent' [0.33]
        ADJP -> 'essential' [0.33]
        PP -> 'to' [0.5]
        PP -> 'in' [0.5]

        # Expansion
        VP_EXP_substance_meronym -> TRANS_SUBSTANCE_MERONYM 'tgt' CONJ COP_SUBSTANCE_MERONYM 'tgt_2' [0.5]
        VP_EXP_substance_meronym -> COP_SUBSTANCE_MERONYM_BASE 'tgt' CONJ TRANS_SUBSTANCE_MERONYM 'tgt_2' [0.5]
        
        CONJ -> 'and' [1.0]
        
        TRANS_SUBSTANCE_MERONYM -> V_S_MERO [1.0]
        V_S_MERO -> 'contains' [0.34] | 'incorporates' [0.33] | 'embeds' [0.33]
        
        COP_SUBSTANCE_MERONYM_BASE -> AUX ADJP PP [1.0]
        COP_SUBSTANCE_MERONYM -> AUX_S_MERO ADJP_S_MERO PP_S_MERO [1.0]
        AUX_S_MERO -> 'is' [0.34] | 'remains' [0.33] | 'appears' [0.33]
        ADJP_S_MERO -> 'composed' [0.34] | 'comprised' [0.33] | 'constituted' [0.33]
        PP_S_MERO -> 'of' [1.0]
    """,

    # ── ConceptNet High Frequency ──────────────────────────────────────────────

    "RelatedTo": """
        VP -> TRANS [0.5]
        VP -> COP [0.5]
        TRANS -> V [1.0]
        V -> 'relates_to' [0.25]
        V -> 'connects_to' [0.25]
        V -> 'links_to' [0.25]
        V -> 'associates_with' [0.25]
        COP -> AUX ADJP PP [1.0]
        AUX -> 'is' [0.34]
        AUX -> 'remains' [0.33]
        AUX -> 'appears' [0.33]
        ADJP -> 'related' [0.34]
        ADJP -> 'connected' [0.33]
        ADJP -> 'linked' [0.33]
        PP -> 'to' [0.5]
        PP -> 'with' [0.5]
    """,

    "HasContext": """
        VP -> TRANS [0.5]
        VP -> COP [0.5]
        TRANS -> V [1.0]
        V -> 'appears_in' [0.25]
        V -> 'occurs_in' [0.25]
        V -> 'belongs_to' [0.25]
        V -> 'originates_in' [0.25]
        COP -> AUX ADJP PP [1.0]
        AUX -> 'is' [0.34]
        AUX -> 'remains' [0.33]
        AUX -> 'appears' [0.33]
        ADJP -> 'used' [0.34]
        ADJP -> 'common' [0.33]
        ADJP -> 'typical' [0.33]
        PP -> 'in' [0.5]
        PP -> 'within' [0.5]
    """,

    "AtLocation": """
        VP -> TRANS [0.5]
        VP -> COP [0.5]
        TRANS -> V [1.0]
        V -> 'inhabits' [0.25]
        V -> 'occupies' [0.25]
        V -> 'frequents' [0.25]
        V -> 'populates' [0.25]
        COP -> AUX ADJP PP [1.0]
        AUX -> 'is' [0.34]
        AUX -> 'remains' [0.33]
        AUX -> 'stays' [0.33]
        ADJP -> 'located' [0.34]
        ADJP -> 'situated' [0.33]
        ADJP -> 'found' [0.33]
        PP -> 'at' [0.34]
        PP -> 'in' [0.33]
        PP -> 'within' [0.33]
        
        # Expansion
        VP_EXP_hypernym -> TRANS_HYPERNYM 'tgt' CONJ VP 'tgt_2' [0.5]
        VP_EXP_hypernym -> COP_HYPERNYM 'tgt' CONJ VP 'tgt_2' [0.5]
    
        VP_EXP_hyponym -> TRANS_HYPONYM 'tgt' CONJ VP 'tgt_2' [0.5]
        VP_EXP_hyponym -> COP_HYPONYM 'tgt' CONJ VP 'tgt_2' [0.5]
        
        CONJ -> 'that' [1.0]
        
        TRANS_HYPERNYM -> V_HYP [1.0]
        V_HYP -> 'exemplifies' [0.25] | 'specializes' [0.25] | 'refines' [0.25] | 'extends' [0.25]
        COP_HYPERNYM -> AUX_HYP ADJP_HYP PP_HYP [1.0]
        AUX_HYP -> 'is' [0.34] | 'remains' [0.33] | 'appears' [0.33]
        ADJP_HYP -> 'subordinate' [0.34] | 'more_specific' [0.33] | 'subsumed' [0.33]
        PP_HYP -> 'to' [0.5] | 'under' [0.5]
        
        TRANS_HYPONYM -> V_HYPO [1.0]
        V_HYPO -> 'generalizes' [0.25] | 'encompasses' [0.25] | 'subsumes' [0.25] | 'categorizes' [0.25]
        COP_HYPONYM -> AUX_HYPO ADJP_HYPO PP_HYPO [1.0]
        AUX_HYPO -> 'is' [0.34] | 'remains' [0.33] | 'appears' [0.33]
        ADJP_HYPO -> 'broader' [0.34] | 'superordinate' [0.33] | 'more_general' [0.33]
        PP_HYPO -> 'than' [0.5] | 'over' [0.5]
    """,

    "UsedFor": """
        VP -> TRANS [0.5]
        VP -> COP [0.5]
        TRANS -> V [1.0]
        V -> 'serves' [0.25]
        V -> 'enables' [0.25]
        V -> 'facilitates' [0.25]
        V -> 'supports' [0.25]
        COP -> AUX ADJP PP [1.0]
        AUX -> 'is' [0.34]
        AUX -> 'remains' [0.33]
        AUX -> 'appears' [0.33]
        ADJP -> 'designed' [0.34]
        ADJP -> 'intended' [0.33]
        ADJP -> 'used' [0.33]
        PP -> 'for' [1.0]

        # Expansion
        VP_EXP_hypernym -> TRANS_HYPERNYM 'tgt' CONJ VP 'tgt_2' [0.5]
        VP_EXP_hypernym -> COP_HYPERNYM 'tgt' CONJ VP 'tgt_2' [0.5]
        
        VP_EXP_hyponym -> TRANS_HYPONYM 'tgt' CONJ VP 'tgt_2' [0.5]
        VP_EXP_hyponym -> COP_HYPONYM 'tgt' CONJ VP 'tgt_2' [0.5]
        
        CONJ -> 'that' [1.0]
        
        TRANS_HYPERNYM -> V_HYP [1.0]
        V_HYP -> 'exemplifies' [0.25] | 'specializes' [0.25] | 'refines' [0.25] | 'extends' [0.25]
        COP_HYPERNYM -> AUX_HYP ADJP_HYP PP_HYP [1.0]
        AUX_HYP -> 'is' [0.34] | 'remains' [0.33] | 'appears' [0.33]
        ADJP_HYP -> 'subordinate' [0.34] | 'more_specific' [0.33] | 'subsumed' [0.33]
        PP_HYP -> 'to' [0.5] | 'under' [0.5]
        
        TRANS_HYPONYM -> V_HYPO [1.0]
        V_HYPO -> 'generalizes' [0.25] | 'encompasses' [0.25] | 'subsumes' [0.25] | 'categorizes' [0.25]
        COP_HYPONYM -> AUX_HYPO ADJP_HYPO PP_HYPO [1.0]
        AUX_HYPO -> 'is' [0.34] | 'remains' [0.33] | 'appears' [0.33]
        ADJP_HYPO -> 'broader' [0.34] | 'superordinate' [0.33] | 'more_general' [0.33]
        PP_HYPO -> 'than' [0.5] | 'over' [0.5]
    """,

    "SimilarTo": """
        VP -> TRANS [0.5]
        VP -> COP [0.5]
        TRANS -> V [1.0]
        V -> 'resembles' [0.34]
        V -> 'mirrors' [0.33]
        V -> 'approximates' [0.33]
        COP -> AUX ADJP PP [1.0]
        AUX -> 'is' [0.34]
        AUX -> 'remains' [0.33]
        AUX -> 'appears' [0.33]
        ADJP -> 'similar' [0.34]
        ADJP -> 'comparable' [0.33]
        ADJP -> 'analogous' [0.33]
        PP -> 'to' [1.0]

        # Expansion
        VP_EXP_DistinctFrom -> TRANS_DIST 'tgt' CONJ VP 'tgt_2' [0.5]
        VP_EXP_DistinctFrom -> COP_DIST 'tgt' CONJ VP 'tgt_2' [0.5]
        
        CONJ -> 'but' [1.0]
        
        TRANS_DIST -> V_DIST [1.0]
        V_DIST -> 'contrasts' [0.34] | 'differs_from' [0.33] | 'diverges_from' [0.33]
        COP_DIST -> AUX_DIST ADJP_DIST PP_DIST [1.0]
        AUX_DIST -> 'is' [0.34] | 'remains' [0.33] | 'appears' [0.33]
        ADJP_DIST -> 'different' [0.34] | 'distinct' [0.33] | 'separate' [0.33]
        PP_DIST -> 'from' [1.0]
    """,

    "Antonym": """
        VP -> TRANS [0.5]
        VP -> COP [0.5]
        TRANS -> V [1.0]
        V -> 'opposes' [0.34]
        V -> 'contradicts' [0.33]
        V -> 'negates' [0.33]
        COP -> AUX ADJP PP [1.0]
        AUX -> 'is' [0.34]
        AUX -> 'remains' [0.33]
        AUX -> 'appears' [0.33]
        ADJP -> 'contrary' [0.34]
        ADJP -> 'opposed' [0.33]
        ADJP -> 'antithetical' [0.33]
        PP -> 'to' [1.0]

        # Expansion
        VP_EXP_SimilarTo -> TRANS_SIM 'tgt' CONJ VP 'tgt_2' [0.5]
        VP_EXP_SimilarTo -> COP_SIM 'tgt' CONJ VP 'tgt_2' [0.5]
        
        CONJ -> 'but' [1.0]
        
        TRANS_SIM -> V_SIM [1.0]
        V_SIM -> 'resembles' [0.34] | 'mirrors' [0.33] | 'approximates' [0.33]
        COP_SIM -> AUX_SIM ADJP_SIM PP_SIM [1.0]
        AUX_SIM -> 'is' [0.34] | 'remains' [0.33] | 'appears' [0.33]
        ADJP_SIM -> 'similar' [0.34] | 'comparable' [0.33] | 'analogous' [0.33]
        PP_SIM -> 'to' [1.0]
    """,

    "CapableOf": """
        VP -> TRANS [0.5]
        VP -> COP [0.5]
        TRANS -> V [1.0]
        V -> 'performs' [0.25]
        V -> 'achieves' [0.25]
        V -> 'executes' [0.25]
        V -> 'accomplishes' [0.25]
        COP -> AUX ADJP PP [1.0]
        AUX -> 'is' [0.34]
        AUX -> 'remains' [0.33]
        AUX -> 'appears' [0.33]
        ADJP -> 'capable' [0.34]
        ADJP -> 'able' [0.33]
        ADJP -> 'suited' [0.33]
        PP -> 'of' [0.5]
        PP -> 'for' [0.5]
        
        # Expansion
        VP_EXP_hypernym -> TRANS_HYPERNYM 'tgt' CONJ VP 'tgt_2' [0.5]
        VP_EXP_hypernym -> COP_HYPERNYM 'tgt' CONJ VP 'tgt_2' [0.5]
        
        VP_EXP_hyponym -> TRANS_HYPONYM 'tgt' CONJ VP 'tgt_2' [0.5]
        VP_EXP_hyponym -> COP_HYPONYM 'tgt' CONJ VP 'tgt_2' [0.5]
        
        CONJ -> 'that' [1.0]
        
        TRANS_HYPERNYM -> V_HYP [1.0]
        V_HYP -> 'exemplifies' [0.25] | 'specializes' [0.25] | 'refines' [0.25] | 'extends' [0.25]
        COP_HYPERNYM -> AUX_HYP ADJP_HYP PP_HYP [1.0]
        AUX_HYP -> 'is' [0.34] | 'remains' [0.33] | 'appears' [0.33]
        ADJP_HYP -> 'subordinate' [0.34] | 'more_specific' [0.33] | 'subsumed' [0.33]
        PP_HYP -> 'to' [0.5] | 'under' [0.5]
        
        TRANS_HYPONYM -> V_HYPO [1.0]
        V_HYPO -> 'generalizes' [0.25] | 'encompasses' [0.25] | 'subsumes' [0.25] | 'categorizes' [0.25]
        COP_HYPONYM -> AUX_HYPO ADJP_HYPO PP_HYPO [1.0]
        AUX_HYPO -> 'is' [0.34] | 'remains' [0.33] | 'appears' [0.33]
        ADJP_HYPO -> 'broader' [0.34] | 'superordinate' [0.33] | 'more_general' [0.33]
        PP_HYPO -> 'than' [0.5] | 'over' [0.5]
    """,

    # ── ConceptNet Medium Frequency ────────────────────────────────────────────

    "HasPrerequisite": """
        VP -> TRANS [0.5]
        VP -> COP [0.5]
        TRANS -> V [1.0]
        V -> 'requires' [0.34]
        V -> 'needs' [0.33]
        V -> 'demands' [0.33]
        COP -> AUX ADJP PP [1.0]
        AUX -> 'is' [0.34]
        AUX -> 'remains' [0.33]
        AUX -> 'appears' [0.33]
        ADJP -> 'dependent' [0.34]
        ADJP -> 'contingent' [0.33]
        ADJP -> 'reliant' [0.33]
        PP -> 'on' [0.5]
        PP -> 'upon' [0.5]
    """,

    "HasProperty": """
        VP -> TRANS [0.5]
        VP -> COP [0.5]
        TRANS -> V [1.0]
        V -> 'exhibits' [0.25]
        V -> 'displays' [0.25]
        V -> 'manifests' [0.25]
        V -> 'shows' [0.25]
        COP -> AUX ADJP PP [1.0]
        AUX -> 'is' [0.34]
        AUX -> 'remains' [0.33]
        AUX -> 'appears' [0.33]
        ADJP -> 'characterized' [0.34]
        ADJP -> 'defined' [0.33]
        ADJP -> 'known' [0.33]
        PP -> 'by' [0.5]
        PP -> 'for' [0.5]

        # Expansion
        VP_EXP_hypernym -> TRANS_HYPERNYM 'tgt' CONJ VP 'tgt_2' [0.5]
        VP_EXP_hypernym -> COP_HYPERNYM 'tgt' CONJ VP 'tgt_2' [0.5]
        
        VP_EXP_hyponym -> TRANS_HYPONYM 'tgt' CONJ VP 'tgt_2' [0.5]
        VP_EXP_hyponym -> COP_HYPONYM 'tgt' CONJ VP 'tgt_2' [0.5]
        
        CONJ_ -> 'that' [1.0]
        
        TRANS_HYPERNYM -> V_HYP [1.0]
        V_HYP -> 'exemplifies' [0.25] | 'specializes' [0.25] | 'refines' [0.25] | 'extends' [0.25]
        COP_HYPERNYM -> AUX_HYP ADJP_HYP PP_HYP [1.0]
        AUX_HYP -> 'is' [0.34] | 'remains' [0.33] | 'appears' [0.33]
        ADJP_HYP -> 'subordinate' [0.34] | 'more_specific' [0.33] | 'subsumed' [0.33]
        PP_HYP -> 'to' [0.5] | 'under' [0.5]
        
        TRANS_HYPONYM -> V_HYPO [1.0]
        V_HYPO -> 'generalizes' [0.25] | 'encompasses' [0.25] | 'subsumes' [0.25] | 'categorizes' [0.25]
        COP_HYPONYM -> AUX_HYPO ADJP_HYPO PP_HYPO [1.0]
        AUX_HYPO -> 'is' [0.34] | 'remains' [0.33] | 'appears' [0.33]
        ADJP_HYPO -> 'broader' [0.34] | 'superordinate' [0.33] | 'more_general' [0.33]
        PP_HYPO -> 'than' [0.5] | 'over' [0.5]
    """,

    "DistinctFrom": """
        VP -> TRANS [0.5]
        VP -> COP [0.5]
        TRANS -> V [1.0]
        V -> 'contrasts' [0.34]
        V -> 'differs_from' [0.33]
        V -> 'diverges_from' [0.33]
        COP -> AUX ADJP PP [1.0]
        AUX -> 'is' [0.34]
        AUX -> 'remains' [0.33]
        AUX -> 'appears' [0.33]
        ADJP -> 'different' [0.34]
        ADJP -> 'distinct' [0.33]
        ADJP -> 'separate' [0.33]
        PP -> 'from' [1.0]
        
        # Expansion
        VP_EXP_Antonym -> TRANS_ANT 'tgt' CONJ VP 'tgt_2' [0.5]
        VP_EXP_Antonym -> COP_ANT 'tgt' CONJ VP 'tgt_2' [0.5]
        
        CONJ -> 'and' [1.0]
        
        TRANS_ANT -> V_ANT [1.0]
        V_ANT -> 'opposes' [0.34] | 'contradicts' [0.33] | 'negates' [0.33]
        COP_ANT -> AUX_ANT ADJP_ANT PP_ANT [1.0]
        AUX_ANT -> 'is' [0.34] | 'remains' [0.33] | 'appears' [0.33]
        ADJP_ANT -> 'contrary' [0.34] | 'opposed' [0.33] | 'antithetical' [0.33]
        PP_ANT -> 'to' [1.0]
    """,

    "HasSubevent": """
        VP -> TRANS [0.5]
        VP -> COP [0.5]
        TRANS -> V [1.0]
        V -> 'involves' [0.34]
        V -> 'includes' [0.33]
        V -> 'encompasses' [0.33]
        COP -> AUX ADJP PP [1.0]
        AUX -> 'is' [0.34]
        AUX -> 'remains' [0.33]
        AUX -> 'appears' [0.33]
        ADJP -> 'characterized' [0.34]
        ADJP -> 'marked' [0.33]
        ADJP -> 'defined' [0.33]
        PP -> 'by' [1.0]
    """,

    "Causes": """
        VP -> TRANS [0.5]
        VP -> COP [0.5]
        TRANS -> V [1.0]
        V -> 'causes' [0.25]
        V -> 'produces' [0.25]
        V -> 'generates' [0.25]
        V -> 'triggers' [0.25]
        COP -> AUX ADJP PP [1.0]
        AUX -> 'is' [0.34]
        AUX -> 'remains' [0.33]
        AUX -> 'appears' [0.33]
        ADJP -> 'responsible' [0.34]
        ADJP -> 'conducive' [0.33]
        ADJP -> 'causal' [0.33]
        PP -> 'for' [0.5]
        PP -> 'to' [0.5]

        # Expansion
        VP_EXP_hypernym -> TRANS_HYPERNYM 'tgt' CONJ VP 'tgt_2' [0.5]
        VP_EXP_hypernym -> COP_HYPERNYM 'tgt' CONJ VP 'tgt_2' [0.5]
        
        VP_EXP_hyponym -> TRANS_HYPONYM 'tgt' CONJ VP 'tgt_2' [0.5]
        VP_EXP_hyponym -> COP_HYPONYM 'tgt' CONJ VP 'tgt_2' [0.5]
        
        CONJ -> 'that' [1.0]
        
        TRANS_HYPERNYM -> V_HYP [1.0]
        V_HYP -> 'exemplifies' [0.25] | 'specializes' [0.25] | 'refines' [0.25] | 'extends' [0.25]
        COP_HYPERNYM -> AUX_HYP ADJP_HYP PP_HYP [1.0]
        AUX_HYP -> 'is' [0.34] | 'remains' [0.33] | 'appears' [0.33]
        ADJP_HYP -> 'subordinate' [0.34] | 'more_specific' [0.33] | 'subsumed' [0.33]
        PP_HYP -> 'to' [0.5] | 'under' [0.5]
        
        TRANS_HYPONYM -> V_HYPO [1.0]
        V_HYPO -> 'generalizes' [0.25] | 'encompasses' [0.25] | 'subsumes' [0.25] | 'categorizes' [0.25]
        COP_HYPONYM -> AUX_HYPO ADJP_HYPO PP_HYPO [1.0]
        AUX_HYPO -> 'is' [0.34] | 'remains' [0.33] | 'appears' [0.33]
        ADJP_HYPO -> 'broader' [0.34] | 'superordinate' [0.33] | 'more_general' [0.33]
        PP_HYPO -> 'than' [0.5] | 'over' [0.5]
    """,

    "MadeOf": """
        VP -> TRANS [0.5]
        VP -> COP [0.5]
        TRANS -> V [1.0]
        V -> 'contains' [0.34]
        V -> 'incorporates' [0.33]
        V -> 'uses' [0.33]
        COP -> AUX ADJP PP [1.0]
        AUX -> 'is' [0.34]
        AUX -> 'remains' [0.33]
        AUX -> 'appears' [0.33]
        ADJP -> 'composed' [0.34]
        ADJP -> 'made' [0.33]
        ADJP -> 'crafted' [0.33]
        PP -> 'of' [0.5]
        PP -> 'from' [0.5]

        # Expansion
        VP_EXP_UsedFor -> TRANS_UF 'tgt' CONJ VP 'tgt_2' [0.5]
        VP_EXP_UsedFor -> COP_UF 'tgt' CONJ VP 'tgt_2' [0.5]

        CONJ -> 'and' [1.0]
        
        TRANS_UF -> V_UF [1.0]
        V_UF -> 'serves' [0.25] | 'enables' [0.25] | 'facilitates' [0.25] | 'supports' [0.25]
        COP_UF -> AUX_UF ADJP_UF PP_UF [1.0]
        AUX_UF -> 'is' [0.34] | 'remains' [0.33] | 'appears' [0.33]
        ADJP_UF -> 'designed' [0.34] | 'intended' [0.33] | 'used' [0.33]
        PP_UF -> 'for' [1.0]
    """,

    "MotivatedByGoal": """
        VP -> TRANS [0.5]
        VP -> COP [0.5]
        TRANS -> V [1.0]
        V -> 'pursues' [0.34]
        V -> 'seeks' [0.33]
        V -> 'targets' [0.33]
        COP -> AUX ADJP PP [1.0]
        AUX -> 'is' [0.34]
        AUX -> 'remains' [0.33]
        AUX -> 'appears' [0.33]
        ADJP -> 'motivated' [0.34]
        ADJP -> 'driven' [0.33]
        ADJP -> 'directed' [0.33]
        PP -> 'by' [0.5]
        PP -> 'toward' [0.5]
    """,

    "ReceivesAction": """
        VP -> TRANS [0.5]
        VP -> COP [0.5]
        TRANS -> V [1.0]
        V -> 'undergoes' [0.34]
        V -> 'receives' [0.33]
        V -> 'accepts' [0.33]
        COP -> AUX ADJP PP [1.0]
        AUX -> 'is' [0.34]
        AUX -> 'remains' [0.33]
        AUX -> 'appears' [0.33]
        ADJP -> 'subject' [0.34]
        ADJP -> 'susceptible' [0.33]
        ADJP -> 'amenable' [0.33]
        PP -> 'to' [1.0]
    """,

    "Desires": """
        VP -> TRANS [0.5]
        VP -> COP [0.5]
        TRANS -> V [1.0]
        V -> 'desires' [0.34]
        V -> 'seeks' [0.33]
        V -> 'craves' [0.33]
        COP -> AUX ADJP PP [1.0]
        AUX -> 'is' [0.34]
        AUX -> 'remains' [0.33]
        AUX -> 'appears' [0.33]
        ADJP -> 'desirous' [0.34]
        ADJP -> 'eager' [0.33]
        ADJP -> 'yearning' [0.33]
        PP -> 'for' [0.5]
        PP -> 'of' [0.5]
        
        # Expansion
        VP_EXP_NotDesires -> TRANS_ND 'tgt' CONJ VP 'tgt_2' [0.5]
        VP_EXP_NotDesires -> COP_ND 'tgt' CONJ VP 'tgt_2' [0.5]
        
        CONJ -> 'but' [1.0]
        
        TRANS_ND -> V_ND [1.0]
        V_ND -> 'avoids' [0.34] | 'rejects' [0.33] | 'refuses' [0.33]
        COP_ND -> AUX_ND ADJP_ND PP_ND [1.0]
        AUX_ND -> 'is' [0.34] | 'remains' [0.33] | 'appears' [0.33]
        ADJP_ND -> 'averse' [0.34] | 'opposed' [0.33] | 'indifferent' [0.33]
        PP_ND -> 'to' [1.0]
    """,

    # ── ConceptNet Low Frequency ───────────────────────────────────────────────

    "NotHasProperty": """
        VP -> TRANS [0.5]
        VP -> COP [0.5]
        TRANS -> V [1.0]
        V -> 'lacks' [0.34]
        V -> 'misses' [0.33]
        V -> 'omits' [0.33]
        COP -> AUX ADJP PP [1.0]
        AUX -> 'is' [0.34]
        AUX -> 'remains' [0.33]
        AUX -> 'appears' [0.33]
        ADJP -> 'devoid' [0.34]
        ADJP -> 'lacking' [0.33]
        ADJP -> 'free' [0.33]
        PP -> 'of' [1.0]
    """,

    "CausesDesire": """
        VP -> TRANS [0.5]
        VP -> COP [0.5]
        TRANS -> V [1.0]
        V -> 'inspires' [0.34]
        V -> 'induces' [0.33]
        V -> 'prompts' [0.33]
        COP -> AUX ADJP PP [1.0]
        AUX -> 'is' [0.34]
        AUX -> 'remains' [0.33]
        AUX -> 'appears' [0.33]
        ADJP -> 'conducive' [0.34]
        ADJP -> 'stimulating' [0.33]
        ADJP -> 'motivating' [0.33]
        PP -> 'to' [0.5]
        PP -> 'toward' [0.5]
        
        # Expansion
        VP_EXP_hypernym -> TRANS_HYPERNYM 'tgt' CONJ VP 'tgt_2' [0.5]
        VP_EXP_hypernym -> COP_HYPERNYM 'tgt' CONJ VP 'tgt_2' [0.5]
        
        VP_EXP_hyponym -> TRANS_HYPONYM 'tgt' CONJ VP 'tgt_2' [0.5]
        VP_EXP_hyponym -> COP_HYPONYM 'tgt' CONJ VP 'tgt_2' [0.5]
        
        CONJ -> 'that' [1.0]
        
        TRANS_HYPERNYM -> V_HYP [1.0]
        V_HYP -> 'exemplifies' [0.25] | 'specializes' [0.25] | 'refines' [0.25] | 'extends' [0.25]
        COP_HYPERNYM -> AUX_HYP ADJP_HYP PP_HYP [1.0]
        AUX_HYP -> 'is' [0.34] | 'remains' [0.33] | 'appears' [0.33]
        ADJP_HYP -> 'subordinate' [0.34] | 'more_specific' [0.33] | 'subsumed' [0.33]
        PP_HYP -> 'to' [0.5] | 'under' [0.5]
        
        TRANS_HYPONYM -> V_HYPO [1.0]
        V_HYPO -> 'generalizes' [0.25] | 'encompasses' [0.25] | 'subsumes' [0.25] | 'categorizes' [0.25]
        COP_HYPONYM -> AUX_HYPO ADJP_HYPO PP_HYPO [1.0]
        AUX_HYPO -> 'is' [0.34] | 'remains' [0.33] | 'appears' [0.33]
        ADJP_HYPO -> 'broader' [0.34] | 'superordinate' [0.33] | 'more_general' [0.33]
        PP_HYPO -> 'than' [0.5] | 'over' [0.5]
    """,

    "HasFirstSubevent": """
        VP -> TRANS [0.5]
        VP -> COP [0.5]
        TRANS -> V [1.0]
        V -> 'begins_with' [0.34]
        V -> 'opens_with' [0.33]
        V -> 'starts_with' [0.33]
        COP -> AUX ADJP PP [1.0]
        AUX -> 'is' [0.34]
        AUX -> 'remains' [0.33]
        AUX -> 'appears' [0.33]
        ADJP -> 'initiated' [0.34]
        ADJP -> 'commenced' [0.33]
        ADJP -> 'launched' [0.33]
        PP -> 'by' [0.5]
        PP -> 'with' [0.5]

        # Expansion
        VP_EXP_HasLastSubevent -> TRANS_LS 'tgt' CONJ VP 'tgt_2' [0.5]
        VP_EXP_HasLastSubevent -> COP_LS 'tgt' CONJ VP 'tgt_2' [0.5]
        
        CONJ -> 'and' [1.0]

        TRANS_LS -> V_LS [1.0]
        V_LS -> 'concludes_with' [0.34] | 'ends_with' [0.33] | 'finishes_with' [0.33]
        COP_LS -> AUX_LS ADJP_LS PP_LS [1.0]
        AUX_LS -> 'is' [0.34] | 'remains' [0.33] | 'appears' [0.33]
        ADJP_LS -> 'concluded' [0.34] | 'finished' [0.33] | 'completed' [0.33]
        PP_LS -> 'by' [0.5] | 'with' [0.5]
    """,

    "HasLastSubevent": """
        VP -> TRANS [0.5]
        VP -> COP [0.5]
        TRANS -> V [1.0]
        V -> 'concludes_with' [0.34]
        V -> 'ends_with' [0.33]
        V -> 'finishes_with' [0.33]
        COP -> AUX ADJP PP [1.0]
        AUX -> 'is' [0.34]
        AUX -> 'remains' [0.33]
        AUX -> 'appears' [0.33]
        ADJP -> 'concluded' [0.34]
        ADJP -> 'finished' [0.33]
        ADJP -> 'completed' [0.33]
        PP -> 'by' [0.5]
        PP -> 'with' [0.5]

        # Expansion
        VP_EXP_HasFirstSubevent -> TRANS_FS 'tgt' CONJ VP 'tgt_2' [0.5]
        VP_EXP_HasFirstSubevent -> COP_FS 'tgt' CONJ VP 'tgt_2' [0.5]
        
        CONJ -> 'and' [1.0]

        TRANS_FS -> V_FS [1.0]
        V_FS -> 'begins_with' [0.34] | 'opens_with' [0.33] | 'starts_with' [0.33]
        COP_FS -> AUX_FS ADJP_FS PP_FS [1.0]
        AUX_FS -> 'is' [0.34] | 'remains' [0.33] | 'appears' [0.33]
        ADJP_FS -> 'initiated' [0.34] | 'commenced' [0.33] | 'launched' [0.33]
        PP_FS -> 'by' [0.5] | 'with' [0.5]
    """,

    "NotDesires": """
        VP -> TRANS [0.5]
        VP -> COP [0.5]
        TRANS -> V [1.0]
        V -> 'avoids' [0.34]
        V -> 'rejects' [0.33]
        V -> 'refuses' [0.33]
        COP -> AUX ADJP PP [1.0]
        AUX -> 'is' [0.34]
        AUX -> 'remains' [0.33]
        AUX -> 'appears' [0.33]
        ADJP -> 'averse' [0.34]
        ADJP -> 'opposed' [0.33]
        ADJP -> 'indifferent' [0.33]
        PP -> 'to' [1.0]
    """,

    "CreatedBy": """
        VP -> COP [1.0]
        COP -> AUX ADJP PP [1.0]
        AUX -> 'is' [0.34]
        AUX -> 'remains' [0.33]
        AUX -> 'appears' [0.33]
        ADJP -> 'created' [0.34]
        ADJP -> 'produced' [0.33]
        ADJP -> 'authored' [0.33]
        PP -> 'by' [1.0]
    """,

    "DefinedAs": """
        VP -> COP [1.0]
        COP -> AUX ADJP PP [1.0]
        AUX -> 'is' [0.34]
        AUX -> 'remains' [0.33]
        AUX -> 'appears' [0.33]
        ADJP -> 'defined' [0.34]
        ADJP -> 'characterized' [0.33]
        ADJP -> 'described' [0.33]
        PP -> 'as' [1.0]
    """,

    "NotCapableOf": """
        VP -> TRANS [0.5]
        VP -> COP [0.5]
        TRANS -> V [1.0]
        V -> 'lacks' [0.34]
        V -> 'avoids' [0.33]
        V -> 'struggles_with' [0.33]
        COP -> AUX ADJP PP [1.0]
        AUX -> 'is' [0.34]
        AUX -> 'remains' [0.33]
        AUX -> 'appears' [0.33]
        ADJP -> 'incapable' [0.34]
        ADJP -> 'unable' [0.33]
        ADJP -> 'unfit' [0.33]
        PP -> 'of' [0.5]
        PP -> 'for' [0.5]
    """,

    "LocatedNear": """
        VP -> TRANS [0.5]
        VP -> COP [0.5]
        TRANS -> V [1.0]
        V -> 'adjoins' [0.34]
        V -> 'borders' [0.33]
        V -> 'neighbors' [0.33]
        COP -> AUX ADJP PP [1.0]
        AUX -> 'is' [0.34]
        AUX -> 'remains' [0.33]
        AUX -> 'appears' [0.33]
        ADJP -> 'near' [0.34]
        ADJP -> 'adjacent' [0.33]
        ADJP -> 'close' [0.33]
        PP -> 'to' [1.0]
    """,
}
