export interface ICD10Code {
  code: string;
  description: string;
  category: string;
}

export const ICD10_CATEGORIES = [
  "Mood Disorders",
  "Anxiety Disorders",
  "Trauma & Stressor-Related",
  "OCD & Related",
  "Bipolar & Related",
  "Personality Disorders",
  "Neurodevelopmental",
  "Eating Disorders",
  "Substance-Related",
  "Other",
] as const;

export const ICD10_CODES: ICD10Code[] = [
  // Mood Disorders
  { code: "F32.0", description: "Major depressive disorder, single episode, mild", category: "Mood Disorders" },
  { code: "F32.1", description: "Major depressive disorder, single episode, moderate", category: "Mood Disorders" },
  { code: "F32.2", description: "Major depressive disorder, single episode, severe without psychotic features", category: "Mood Disorders" },
  { code: "F32.9", description: "Major depressive disorder, single episode, unspecified", category: "Mood Disorders" },
  { code: "F33.0", description: "Major depressive disorder, recurrent, mild", category: "Mood Disorders" },
  { code: "F33.1", description: "Major depressive disorder, recurrent, moderate", category: "Mood Disorders" },
  { code: "F33.2", description: "Major depressive disorder, recurrent, severe without psychotic features", category: "Mood Disorders" },
  { code: "F34.1", description: "Dysthymic disorder (persistent depressive disorder)", category: "Mood Disorders" },

  // Anxiety Disorders
  { code: "F41.0", description: "Panic disorder", category: "Anxiety Disorders" },
  { code: "F41.1", description: "Generalized anxiety disorder", category: "Anxiety Disorders" },
  { code: "F40.10", description: "Social anxiety disorder (social phobia)", category: "Anxiety Disorders" },
  { code: "F40.00", description: "Agoraphobia, unspecified", category: "Anxiety Disorders" },
  { code: "F40.218", description: "Other animal type phobia", category: "Anxiety Disorders" },
  { code: "F41.8", description: "Other specified anxiety disorders", category: "Anxiety Disorders" },
  { code: "F41.9", description: "Anxiety disorder, unspecified", category: "Anxiety Disorders" },

  // Trauma & Stressor-Related
  { code: "F43.10", description: "Post-traumatic stress disorder, unspecified", category: "Trauma & Stressor-Related" },
  { code: "F43.11", description: "Post-traumatic stress disorder, acute", category: "Trauma & Stressor-Related" },
  { code: "F43.12", description: "Post-traumatic stress disorder, chronic", category: "Trauma & Stressor-Related" },
  { code: "F43.0", description: "Acute stress reaction", category: "Trauma & Stressor-Related" },
  { code: "F43.20", description: "Adjustment disorder, unspecified", category: "Trauma & Stressor-Related" },
  { code: "F43.21", description: "Adjustment disorder with depressed mood", category: "Trauma & Stressor-Related" },
  { code: "F43.22", description: "Adjustment disorder with anxiety", category: "Trauma & Stressor-Related" },
  { code: "F43.23", description: "Adjustment disorder with mixed anxiety and depressed mood", category: "Trauma & Stressor-Related" },
  { code: "F43.25", description: "Adjustment disorder with mixed disturbance of emotions and conduct", category: "Trauma & Stressor-Related" },

  // OCD & Related
  { code: "F42.2", description: "Mixed obsessional thoughts and acts", category: "OCD & Related" },
  { code: "F42.0", description: "Predominantly obsessional thoughts or ruminations", category: "OCD & Related" },
  { code: "F42.1", description: "Predominantly compulsive acts", category: "OCD & Related" },
  { code: "F45.22", description: "Body dysmorphic disorder", category: "OCD & Related" },

  // Bipolar & Related
  { code: "F31.0", description: "Bipolar disorder, current episode hypomanic", category: "Bipolar & Related" },
  { code: "F31.31", description: "Bipolar disorder, current episode depressed, mild", category: "Bipolar & Related" },
  { code: "F31.32", description: "Bipolar disorder, current episode depressed, moderate", category: "Bipolar & Related" },
  { code: "F31.81", description: "Bipolar II disorder", category: "Bipolar & Related" },
  { code: "F34.0", description: "Cyclothymic disorder", category: "Bipolar & Related" },

  // Personality Disorders
  { code: "F60.3", description: "Borderline personality disorder", category: "Personality Disorders" },
  { code: "F60.4", description: "Histrionic personality disorder", category: "Personality Disorders" },
  { code: "F60.5", description: "Obsessive-compulsive personality disorder", category: "Personality Disorders" },
  { code: "F60.6", description: "Avoidant personality disorder", category: "Personality Disorders" },

  // Neurodevelopmental
  { code: "F90.0", description: "ADHD, predominantly inattentive type", category: "Neurodevelopmental" },
  { code: "F90.1", description: "ADHD, predominantly hyperactive type", category: "Neurodevelopmental" },
  { code: "F90.2", description: "ADHD, combined type", category: "Neurodevelopmental" },
  { code: "F84.0", description: "Autistic disorder", category: "Neurodevelopmental" },

  // Eating Disorders
  { code: "F50.00", description: "Anorexia nervosa, unspecified", category: "Eating Disorders" },
  { code: "F50.2", description: "Bulimia nervosa", category: "Eating Disorders" },
  { code: "F50.81", description: "Binge eating disorder", category: "Eating Disorders" },

  // Substance-Related
  { code: "F10.20", description: "Alcohol use disorder, moderate", category: "Substance-Related" },
  { code: "F12.20", description: "Cannabis use disorder, moderate", category: "Substance-Related" },
  { code: "F11.20", description: "Opioid use disorder, moderate", category: "Substance-Related" },

  // Other
  { code: "F44.9", description: "Dissociative disorder, unspecified", category: "Other" },
  { code: "F48.1", description: "Depersonalization-derealization syndrome", category: "Other" },
  { code: "F45.1", description: "Undifferentiated somatoform disorder", category: "Other" },
  { code: "F51.01", description: "Primary insomnia", category: "Other" },
  { code: "Z63.0", description: "Problems in relationship with spouse or partner", category: "Other" },
];
