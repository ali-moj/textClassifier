package com.ali.textClassifier.utils.nlp;


import com.ali.textClassifier.model.Term;
import com.google.common.collect.Sets;

import java.util.Set;

public class TermFilter {
    private final static Set<String> POSSet = Sets.newHashSet("w", "nx", "m", "t", "nt");

    public static boolean filter(Term term) {
        return POSSet.stream().noneMatch(term.getPOS()::startsWith)
                && !NLPTools.instance().isStopWord(term.getWord()) && !term.getWord().matches("^\\d+(.*)");
    }
}