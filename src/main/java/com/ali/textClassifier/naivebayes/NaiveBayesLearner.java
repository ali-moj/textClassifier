package com.ali.textClassifier.naivebayes;

import com.ali.textClassifier.feature.ChiSquaredStrategy;
import com.ali.textClassifier.feature.Feature;
import com.ali.textClassifier.feature.FeatureSelection;
import com.ali.textClassifier.model.Category;
import com.ali.textClassifier.model.Term;
import com.ali.textClassifier.model.TrainSet;
import com.google.common.collect.Sets;
import lombok.extern.slf4j.Slf4j;

import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.function.Function;

import static com.ali.textClassifier.naivebayes.NaiveBayesModels.Bernoulli;
import static com.ali.textClassifier.naivebayes.NaiveBayesModels.Multinomial;
import static java.util.stream.Collectors.*;

@Slf4j
public class NaiveBayesLearner {

    private int total;

    private NaiveBayesKnowledgeBase knowledgeBase;

    interface Model {

        double Pprior(int total, final Category category);


        double Pcondition(final Feature feature, final Category category, double smoothing);
    }

    private Model model;

    private Set<Category> categorySet;
    private Set<Feature> featureSet;

    private TrainSet trainSet;

    public NaiveBayesLearner(Model model, TrainSet trainSet, Set<Feature> selectedFeatures) {
        this.model = model;
        this.trainSet = trainSet;
        this.featureSet = selectedFeatures;
        this.knowledgeBase = new NaiveBayesKnowledgeBase();
    }

    public NaiveBayesLearner statistics() {
        log.info("开始统计...");
        this.total = total();
        log.info("total : " + total);
        this.categorySet = trainSet.getCategorySet();
        featureSet.forEach(f -> f.getCategoryTermCounter().forEach((category, count) -> category.setTermCount(category.getTermCount() + count)));
        categorySet.stream().map(Category::toString).forEach(log::info);
        return this;
    }

    public NaiveBayesKnowledgeBase build() {
        this.knowledgeBase.setCategories(createCategorySummaries(categorySet));
        this.knowledgeBase.setFeatures(createFeatureSummaries(featureSet, categorySet));
        return knowledgeBase;
    }

    private Map<String, NaiveBayesKnowledgeBase.FeatureSummary> createFeatureSummaries(final Set<Feature> featureSet, final Set<Category> categorySet) {
        return featureSet.parallelStream()
                .map(f -> knowledgeBase.createFeatureSummary(f, getPconditions(f, categorySet)))
                .collect(toMap(NaiveBayesKnowledgeBase.FeatureSummary::getWord, Function.identity()));
    }

    private Map<String, Double> createCategorySummaries(final Set<Category> categorySet) {
        return categorySet.stream().collect(toMap(Category::getName, c -> model.Pprior(total, c)));
    }

    private Map<String, Double> getPconditions(final Feature feature, final Set<Category> categorySet) {
        final double smoothing = smoothing();
        return categorySet.stream()
                .collect(toMap(Category::getName, c -> model.Pcondition(feature, c, smoothing)));
    }

    private int total() {
        if (model == Multinomial)
            return featureSet.parallelStream()
                    .map(Feature::getTerm)
                    .mapToInt(Term::getTf)
                    .sum();
        else if (model == Bernoulli)
            return trainSet.getTotalDoc();
        return 0;
    }

    private double smoothing() {
        if (model == Multinomial)
            return this.featureSet.size();
        else if (model == Bernoulli)
            return 2.0;
        return 0.0;
    }


    public static void main(String[] args) {
        TrainSet trainSet = new TrainSet(System.getProperty("user.dir") + "/trainset/");

        FeatureSelection featureSelection = new FeatureSelection(new ChiSquaredStrategy(trainSet.getCategorySet(), trainSet.getTotalDoc()));
        List<Feature> features = featureSelection.select(trainSet.getDocs());
        log.info(">>>>>:[" + features.size() + "]");
        features.forEach(System.out::println);

        NaiveBayesModels model = Multinomial;
        NaiveBayesLearner learner = new NaiveBayesLearner(model, trainSet, Sets.newHashSet(features));
        learner.statistics().build().write(model.getModelPath());
        log.info(">>>>:" + model.getModelPath());
    }

}
