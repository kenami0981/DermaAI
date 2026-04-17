using Microsoft.EntityFrameworkCore.Migrations;

#nullable disable

namespace Derma.Infrastructure.Migrations
{
    /// <inheritdoc />
    public partial class AddSkinAnalysis : Migration
    {
        /// <inheritdoc />
        protected override void Up(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.RenameColumn(
                name: "Notes",
                table: "SkinAnalysis",
                newName: "notes");

            migrationBuilder.AlterColumn<double>(
                name: "confidence",
                table: "SkinAnalysis",
                type: "REAL",
                nullable: true,
                oldClrType: typeof(double),
                oldType: "REAL");
        }

        /// <inheritdoc />
        protected override void Down(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.RenameColumn(
                name: "notes",
                table: "SkinAnalysis",
                newName: "Notes");

            migrationBuilder.AlterColumn<double>(
                name: "confidence",
                table: "SkinAnalysis",
                type: "REAL",
                nullable: false,
                defaultValue: 0.0,
                oldClrType: typeof(double),
                oldType: "REAL",
                oldNullable: true);
        }
    }
}
