package com.ali.textClassifier.model;

import com.google.common.collect.Maps;
import lombok.*;

import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;

@Data
@EqualsAndHashCode(of = {"word"})
public class Term {

    private final String word;

    private final String POS;

    private int tf;

    public Term(String word, String POS, int tf) {
        this.word = word.toLowerCase();
        this.POS = POS;
        this.tf = tf;
    }

}
