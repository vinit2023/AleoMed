"use client";
import { FC } from "react";
import { Article } from "@/components/Article/Article";
import { Step, Stepper } from "@/components/Stepper/Stepper";
import { Button } from "@radix-ui/themes";
import steps from "@/mocks/steps.json";
import Link from "next/link";

type ResultLoaderProps = {};

export const ResultLoader: FC<ResultLoaderProps> = () => {
  const stepsList: Step[] = steps;

  return (
    <Article
      title="Your data was submitted"
      beforeArticle={
        <Stepper
          steps={stepsList}
          currentStep="done"
          passedSteps={[
            "connect-wallet",
            "participate",
            "provide-personal-information",
          ]}
        />
      }
      backUrl="/"
      isProtected
    >
      <div style={{ textAlign: "center" }}>
        <div>
          <svg
            width="158"
            height="158"
            viewBox="0 0 158 158"
            fill="none"
            xmlns="http://www.w3.org/2000/svg"
          >
            <circle cx="79" cy="79" r="79" fill="#F1FCF0" />
            <path
              fillRule="evenodd"
              clipRule="evenodd"
              d="M117.285 39.5943C118.337 40.488 118.466 42.0656 117.572 43.1181L82.4131 84.5279C82.3106 84.6487 82.2032 84.7652 82.0912 84.8772C79.8131 87.1552 76.1197 87.1552 73.8416 84.8772L63.8989 74.9345C62.9226 73.9582 62.9226 72.3752 63.8989 71.3989C64.8752 70.4226 66.4581 70.4226 67.4344 71.3989L77.3772 81.3416C77.7026 81.6671 78.2302 81.6671 78.5557 81.3416C78.5717 81.3256 78.587 81.309 78.6017 81.2918L113.761 39.882C114.655 38.8295 116.232 38.7007 117.285 39.5943ZM39 79C39 56.9086 56.9086 39.0001 79 39.0001C84.4188 39.0001 89.5919 40.0789 94.3115 42.036C95.5869 42.5648 96.1921 44.0275 95.6633 45.3029C95.1344 46.5783 93.6717 47.1835 92.3963 46.6546C88.2732 44.9449 83.7501 44 79 44C59.67 44 44 59.6701 44 79C44 98.33 59.67 114 79 114C98.33 114 114 98.33 114 79C114 74.2499 113.055 69.7268 111.345 65.6037C110.817 64.3283 111.422 62.8656 112.697 62.3368C113.973 61.8079 115.435 62.4131 115.964 63.6885C117.921 68.4081 119 73.5812 119 79C119 101.091 101.091 119 79 119C56.9086 119 39 101.091 39 79Z"
              fill="#40E357"
            />
          </svg>
        </div>
        <h1 style={{ fontSize: "32px" }}>Thank you!</h1>
        <Link href="/">
          <Button variant="solid" radius="full" size="4">
            See more researches
          </Button>
        </Link>
      </div>
    </Article>
  );
};
