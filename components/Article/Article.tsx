import { FC, PropsWithChildren, ReactNode } from "react";
import cn from "classnames";
import Link from "next/link";
import { Button } from "@radix-ui/themes";
import { ArrowCircleLeft } from "@phosphor-icons/react";
import { ProtectContent } from "@/components/ProtectContent/ProtectContent";
import styles from "./Article.module.scss";

type ArticleProps = {
  className?: string;
  title?: string;
  description?: string;
  backUrl?: string;
  buttonUrl?: string;
  buttonText?: string;
  beforeArticle?: ReactNode;
  isProtected?: boolean;
  onClick?: () => void;
};

export const Article: FC<PropsWithChildren<ArticleProps>> = ({
  className,
  title,
  description,
  backUrl,
  buttonUrl,
  buttonText,
  beforeArticle,
  children,
  isProtected,
  onClick,
}) => {
  return (
    <article className={cn(styles.article, className)}>
      {beforeArticle && (
        <div className={styles.beforeArticle}>{beforeArticle}</div>
      )}

      {(title || (buttonText && onClick)) && (
        <header className={styles.header}>
          {backUrl && (
            <Link href={backUrl} className={styles.back}>
              <ArrowCircleLeft />
            </Link>
          )}

          <div className={styles.text}>
            {title && <h1 className={styles.title}>{title}</h1>}
            {description && <p className={styles.description}>{description}</p>}
          </div>

          {buttonText && (
            <div className={styles.actions}>
              {buttonUrl && !onClick && (
                <Link href={buttonUrl} className={styles.action}>
                  <Button>{buttonText}</Button>
                </Link>
              )}
              {onClick && !buttonUrl && (
                <Button onClick={onClick}>{buttonText}</Button>
              )}
            </div>
          )}
        </header>
      )}
      <section className={styles.content}>
        {isProtected ? <ProtectContent>{children}</ProtectContent> : children}
      </section>
    </article>
  );
};
