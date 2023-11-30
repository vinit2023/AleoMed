"use client";
import { FC, useCallback, useMemo, useState } from "react";
import { useRouter } from "next/navigation";
import { Article } from "@/components/Article/Article";
import {
  CreateResearchForm,
  NewResearch,
} from "@/features/researcher/CreateResearchForm/CreateResearchForm";
import { ProgressActions } from "@/components/ProgressActions/ProgressActions";
import { LoaderOverflow } from "@/components/LoaderOverflow/LoaderOverflow";

type ResearchCreateLoaderProps = {};

export const ResearchCreateLoader: FC<ResearchCreateLoaderProps> = () => {
  const router = useRouter();
  const [researchData, setResearchData] = useState<NewResearch | null>(null);

  const [isLoading, setIsLoading] = useState(false);

  const isDataValid = useMemo(() => {
    return (
      !!researchData?.title &&
      researchData?.title?.length >= 5 &&
      !!researchData?.files &&
      researchData?.files?.length > 0
    );
  }, [researchData?.files, researchData?.title]);

  const handleSubmit = useCallback(async () => {
    setIsLoading(true);
    if (!researchData) return;
    const formData = new FormData();
    formData.append("title", researchData.title);
    if (researchData.description) {
      formData.append("description", researchData.description);
    }
    formData.append("files", researchData.files[0]);
    formData.append("draft", researchData.draft.toString());
    formData.append("contractName", researchData?.files?.[0]?.name);

    try {
      const res = await fetch("/upload", {
        method: "POST",
        body: formData,
      });

      const resData = await res.json();

      console.log(resData);
      router.push(`/researchers/${resData.id}`);
    } catch (err) {
      console.log(err);
    }
    // console.log("TAKE MY NEW RESEARCH!", researchData);
    setTimeout(() => {
      setIsLoading(false);
    }, 2000);
  }, [researchData, router]);

  return (
    <Article
      title="Create new research"
      description="Fill in Your Details"
      backUrl="/"
      isProtected
    >
      <CreateResearchForm onChange={setResearchData} />
      <ProgressActions canNext={isDataValid} onSubmit={handleSubmit} />

      {isLoading && <LoaderOverflow title="Sending..." />}
    </Article>
  );
};
